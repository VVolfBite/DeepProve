use super::{ChallengeStorage, Context, Proof, TableProof};
use crate::{
    Claim, Element, VectorTranscript,
    commit::{
        compute_betas_eval,
        context::{self, PolyId},
    },
    layers::{
        LayerProof,
        provable::{NodeId, OpInfo, ProvableOp},
    },
    lookup::{
        context::{LookupWitness, generate_lookup_witnesses},
        logup_gkr::prover::batch_prove as logup_batch_prove,
        witness::LogUpWitness,
    },
    model::{InferenceStep, InferenceTrace, ToIterator},
    tensor::get_root_of_unity,
};
use anyhow::{anyhow, ensure};
use ff_ext::ExtensionField;
use std::collections::HashMap;
use std::{env, fs, path::PathBuf, time::{SystemTime, UNIX_EPOCH}};
use rmp_serde;
use tracing::trace;

use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::VirtualPolynomial,
};
use serde::{Serialize, de::DeserializeOwned};

use sumcheck::structs::IOPProverState;
use timed::timed_instrument;
use tracing::debug;
use transcript::Transcript;
use utils::{Metrics, stream_metrics};

/// Prover generates a series of sumcheck proofs to prove the inference of a model
pub struct Prover<'a, E: ExtensionField, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    ctx: &'a Context<E, PCS>,
    // proofs for each layer being filled
    proofs: HashMap<NodeId, LayerProof<E, PCS>>,
    table_proofs: Vec<TableProof<E, PCS>>,
    pub(crate) transcript: &'a mut T,
    /// Proves commitment openings
    pub(crate) commit_prover: context::CommitmentProver<E, PCS>,
    /// The lookup witnesses
    pub(crate) lookup_witness: HashMap<NodeId, Vec<LogUpWitness<E, PCS>>>,
    /// The Lookup table witness
    pub(crate) table_witness: Vec<LogUpWitness<E, PCS>>,
    /// Stores all the challenges for the different lookup/table types
    pub(crate) challenge_storage: ChallengeStorage<E>,
}

pub struct BatchFFTProof<E: ExtensionField> {
    pub proof: sumcheck::structs::IOPProof<E>,
    pub claims: Vec<E>,
    pub matrix_eval: (Vec<sumcheck::structs::IOPProof<E>>, Vec<Vec<E>>),
}

impl<'a, E, T, PCS> Prover<'a, E, T, PCS>
where
    T: Transcript<E>,
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub fn new(ctx: &'a Context<E, PCS>, transcript: &'a mut T) -> Self {
        Self {
            ctx,
            transcript,
            proofs: Default::default(),
            table_proofs: Vec::default(),
            commit_prover: context::CommitmentProver::<E, PCS>::new(),
            lookup_witness: HashMap::default(),
            table_witness: Vec::default(),
            challenge_storage: ChallengeStorage::default(),
        }
    }

    pub(crate) fn add_common_claims(
        &mut self,
        node_id: NodeId,
        claims: HashMap<PolyId, Claim<E>>,
    ) -> anyhow::Result<()> {
        self.commit_prover
            .add_common_claims(&self.ctx.commitment_ctx, node_id, claims)
    }

    pub(crate) fn lookup_witness(
        &mut self,
        id: NodeId,
    ) -> anyhow::Result<Vec<LogUpWitness<E, PCS>>> {
        self.lookup_witness
            .remove(&id)
            .ok_or(anyhow!("No lookup witness found for node {id}!"))
    }

    pub(crate) fn push_proof(&mut self, node_id: NodeId, proof: LayerProof<E, PCS>) {
        // Always persist intermediate layer proof to ./proofs with timestamped filename.
        let _ = Self::save_intermediate_proof(node_id, &proof);
        self.proofs.insert(node_id, proof);
    }

    fn save_intermediate_proof(node_id: NodeId, proof: &LayerProof<E, PCS>) -> Result<PathBuf, ()> {
        // Choose directory: env var if set, else default to ./proofs
        let dir = env::var("ZKML_INTERMEDIATE_DIR").unwrap_or_else(|_| "proofs".to_string());
        let variant = proof.variant_name().to_lowercase();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|_| ())?.as_millis();
        let path: PathBuf = PathBuf::from(dir).join(format!(
            "layer-{}-{:06}-{}.rmp",
            now, node_id, variant
        ));
        if let Some(parent) = path.parent() {
            if let Err(_e) = fs::create_dir_all(parent) {
                return Err(());
            }
        }
        match rmp_serde::to_vec_named(proof) {
            Ok(bytes) => fs::write(&path, bytes).map_err(|_| ()).map(|_| path),
            Err(_e) => Err(()),
        }
    }

    #[timed::timed_instrument(level = "debug")]
    fn prove_tables(&mut self) -> anyhow::Result<()> {
        let mut table_witness = std::mem::replace(&mut self.table_witness, Vec::new());
        table_witness.reverse();

        while let Some(table_witness) = table_witness.pop() {
            let logup_input = table_witness.get_logup_input(&self.challenge_storage)?;
            let table_type = table_witness.table_type();
            let mut comm_with_wit = table_witness.into_commitments();
            ensure!(
                comm_with_wit.len() == 1,
                "Table witness should have one commitment"
            );
            let comm_with_wit = comm_with_wit.pop().expect("Length was checked above");
            let multiplicity_commit = PCS::get_pure_commitment(&comm_with_wit.0);

            // Make the proof for the table
            let table_proof = logup_batch_prove(&logup_input, self.transcript)?;

            // Add the multiplicity poly claim
            self.commit_prover.add_witness_claim(
                comm_with_wit,
                table_proof.output_claims().first().unwrap().clone(),
            )?;

            // Add any table poly claims to the commitment prover
            let table_poly_claims = table_type.table_claims(table_proof.output_claims());

            if !table_poly_claims.is_empty() {
                // If the table poly claims aren't empty there should only be 1
                ensure!(
                    table_poly_claims.len() == 1,
                    "If table poly claims isn't empty we should only have 1, got: {}",
                    table_poly_claims.len()
                );
                self.commit_prover.add_table_claim(
                    &self.ctx.commitment_ctx,
                    table_type,
                    table_poly_claims[0].clone(),
                )?;
            }

            self.table_proofs.push(TableProof {
                multiplicity_commit,
                lookup: table_proof,
            });
        }
        Ok(())
    }

    // Protocol for proving the correct computation of the FFT/iFFT matrix.
    // For more details look at the zkCNN paper.
    // F_middle : all intermediate evaluations retrieved by the phiGinit algorithm
    // r1: the initial random point used to reduce the matrix into vector
    // r2: the random point produced by the sumcheck
    pub fn delegate_matrix_evaluation(
        &mut self,
        f_middle: &mut [Vec<E>],
        r1: Vec<E>,
        mut r2: Vec<E>,
        is_fft: bool,
    ) -> (Vec<sumcheck::structs::IOPProof<E>>, Vec<Vec<E>>) {
        let mut omegas = vec![E::ZERO; 1 << r1.len()];
        self.phi_pow_init(&mut omegas, r1.len(), is_fft);

        let mut proofs: Vec<sumcheck::structs::IOPProof<E>> = Vec::new();
        let mut claims: Vec<Vec<E>> = Vec::new();

        for l in (0..(r1.len() - 1)).rev() {
            let mut phi = vec![E::ZERO; f_middle[l].len()];
            let beta = compute_betas_eval(&r2[0..(r2.len() - 1)]);

            for i in 0..(phi.len()) {
                if !is_fft && l == f_middle.len() - 1 {
                    phi[i] = (E::ONE - r2[r2.len() - 1])
                        * (E::ONE - r1[(f_middle.len() - 1) - l]
                            + r1[(f_middle.len() - 1) - l]
                                * omegas[i << ((f_middle.len() - 1) - l)]);
                } else {
                    phi[i] = E::ONE - r1[(f_middle.len() - 1) - l]
                        + (E::ONE - E::from_canonical_u64(2) * r2[r2.len() - 1])
                            * r1[(f_middle.len() - 1) - l]
                            * omegas[i << ((f_middle.len() - 1) - l)];
                }
            }

            let f1 = beta.into_mle();
            let f2 = phi.into_mle();
            let f3 = f_middle[l].clone().into_mle();

            let mut vp = VirtualPolynomial::<E>::new(f1.num_vars);
            vp.add_mle_list(
                vec![f1.clone().into(), f2.clone().into(), f3.clone().into()],
                E::ONE,
            );
            #[allow(deprecated)]
            let (proof, state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);
            let claim: Vec<E> = state.get_mle_final_evaluations();
            r2 = proof.point.clone();
            proofs.push(proof);
            claims.push(claim);
        }
        (proofs, claims)
    }

    // Compute powers of roots of unity
    pub fn phi_pow_init(&mut self, phi_mul: &mut [E], n: usize, is_fft: bool) {
        let length = 1 << n;
        let rou: E = get_root_of_unity(n);

        let mut phi = rou;
        if is_fft {
            phi = phi.inverse();
        }
        phi_mul[0] = E::ONE;
        for i in 1..length {
            phi_mul[i] = phi_mul[i - 1] * phi;
        }
    }

    // Efficiently compute the omegas of FFT/iFFT matrix reduced at rx
    // This is a copy-paste implementation from zkCNN paper
    pub fn phi_g_init(
        &mut self,
        phi_g: &mut [E],
        mid_phi_g: &mut [Vec<E>],
        rx: Vec<E>,
        scale: E,
        n: usize,
        is_fft: bool,
    ) {
        let mut phi_mul = vec![E::ZERO; 1 << n];
        self.phi_pow_init(&mut phi_mul, n, is_fft);
        if is_fft {
            phi_g[0] = scale;
            phi_g[1] = scale;
            for i in 1..(n + 1) {
                for b in 0..(1 << (i - 1)) {
                    let l = b;
                    let r = b ^ (1 << (i - 1));
                    let m = n - i;
                    let tmp1 = E::ONE - rx[m];
                    let tmp2 = rx[m] * phi_mul[b << m];
                    phi_g[r] = phi_g[l] * (tmp1 - tmp2);
                    phi_g[l] *= tmp1 + tmp2;
                }
                if i < n {
                    mid_phi_g[i - 1] = vec![E::ZERO; 1 << (i)];
                    mid_phi_g[i - 1][..(1 << (i))].copy_from_slice(&phi_g[..(1 << (i))]);
                }
            }
        } else {
            phi_g[0] = scale;
            for i in 1..n {
                for b in 0..(1 << (i - 1)) {
                    let l = b;
                    let r = b ^ (1 << (i - 1));
                    let m = n - i;

                    let tmp1 = E::ONE - rx[m];
                    let tmp2 = rx[m] * phi_mul[b << m];
                    // printf("%d,%d\n",r,l );
                    phi_g[r] = phi_g[l] * (tmp1 - tmp2);
                    phi_g[l] *= tmp1 + tmp2;
                }
                mid_phi_g[i - 1] = vec![E::ZERO; 1 << i];
                mid_phi_g[i - 1][..(1 << (i))].copy_from_slice(&phi_g[..(1 << (i))]);
            }
            for (b, item) in phi_mul.iter().enumerate().take(1 << (n - 1)) {
                let l = b;
                let tmp1 = E::ONE - rx[0];
                let tmp2 = rx[0] * *item;
                phi_g[l] *= tmp1 + tmp2;
            }
        }
    }
    // The prove_batch_fft and prove_batch_ifft are extensions of prove_fft and prove_ifft but in the batch setting.
    // Namely when we want to proof fft or ifft for MORE THAN ONE INSTANCES.
    // In particular, instead of proving y = Wx we want to prove Y = WX where Y,X are matrixes.
    // Following the matrix to matrix multiplication protocol, let y_eval = Y(r1,r2).
    // Then we want to prove a sumcheck instance of the form y_eval = sum_{i \in [n]}W(r1,i)X(i,r2).
    pub fn prove_batch_fft(&mut self, r: Vec<E>, x: &mut [Vec<E>]) -> BatchFFTProof<E> {
        let padded_rows = 2 * x[0].len();
        for item in x.iter_mut() {
            item.resize(padded_rows, E::ZERO);
        }
        // Partition r in (r1,r2)
        let mut r1 = vec![E::ZERO; x[0].len().ilog2() as usize];
        let mut r2 = vec![E::ZERO; x.len().ilog2() as usize];
        let r1_len = r1.len();
        r1.copy_from_slice(&r[..r1_len]);

        for i in 0..r2.len() {
            r2[i] = r[i + r1.len()];
        }
        // compute W(r1,i)
        let mut w_red: Vec<E> = vec![E::ZERO; x[0].len()];
        let mut f_middle: Vec<Vec<E>> = vec![Vec::new(); r1.len() - 1];
        self.phi_g_init(
            &mut w_red,
            &mut f_middle,
            r1.clone(),
            E::ONE,
            x[0].len().ilog2() as usize,
            false,
        );
        // compute X(i,r2)

        let mut f_m = x.iter().flatten().cloned().collect::<Vec<_>>().into_mle();

        f_m.fix_high_variables_in_place(&r2);

        // Construct the virtual polynomial and run the sumcheck prover
        let f_red = w_red.into_mle();

        let mut vp = VirtualPolynomial::<E>::new(f_m.num_vars);
        vp.add_mle_list(vec![f_m.clone().into(), f_red.clone().into()], E::ONE);
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);

        let claims = state.get_mle_final_evaluations();
        let out_point = proof.point.clone();
        BatchFFTProof {
            proof,
            claims,
            matrix_eval: self.delegate_matrix_evaluation(
                &mut f_middle,
                r1.clone(),
                out_point,
                false,
            ),
        }
    }

    pub fn prove_batch_ifft(&mut self, r: Vec<E>, prod: &[Vec<E>]) -> BatchFFTProof<E> {
        let scale: E = E::from_canonical_u64(prod[0].len() as u64).inverse();

        // Partition r in (r1,r2)
        let mut r1 = vec![E::ZERO; prod[0].len().ilog2() as usize];
        let mut r2 = vec![E::ZERO; prod.len().ilog2() as usize];
        let r1_len = r1.len();
        r1.copy_from_slice(&r[..r1_len]);
        assert_eq!(
            r1[r1.len() - 1],
            E::ZERO,
            "Error in randomness init batch ifft"
        );
        for i in 0..r2.len() {
            r2[i] = r[i + r1.len()];
        }
        // compute W(r1,i)
        let mut w_red: Vec<E> = vec![E::ZERO; prod[0].len()];
        let mut f_middle: Vec<Vec<E>> = vec![Vec::new(); r1.len() - 1];
        self.phi_g_init(
            &mut w_red,
            &mut f_middle,
            r1.clone(),
            scale,
            prod[0].len().ilog2() as usize,
            true,
        );
        let f_red = w_red.into_mle();
        // compute X(i,r2)
        let mut f_m = prod
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>()
            .into_mle();
        f_m.fix_high_variables_in_place(&r2);

        // Construct the virtual polynomial and run the sumcheck prover
        let mut vp = VirtualPolynomial::<E>::new(f_m.num_vars);
        vp.add_mle_list(vec![f_m.clone().into(), f_red.clone().into()], E::ONE);
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, self.transcript);

        let claims = state.get_mle_final_evaluations();

        let out_point = proof.point.clone();
        BatchFFTProof {
            proof,
            claims,
            matrix_eval: self.delegate_matrix_evaluation(
                &mut f_middle,
                r1.clone(),
                out_point,
                true,
            ),
        }
    }

    pub fn prove<'b>(
        mut self,
        full_trace: InferenceTrace<'b, E, Element>,
    ) -> anyhow::Result<Proof<E, PCS>> {
        debug!("== Instantiate witness context ==");

        let metrics = Metrics::new();
        self.ctx.write_to_transcript(self.transcript)?;
        self.instantiate_witness_ctx(&full_trace)?;

        let span = metrics.to_span();
        stream_metrics("Witness context", &span);
        debug!("== Witness context metrics {} ==", span);

        debug!("== Generating claims ==");
        let metrics = Metrics::new();
        let trace = full_trace.into_fields();
        // this is the random set of variables to fix at each step derived as the output of
        // sumcheck.
        // For the first step, so before the first sumcheck, we generate it from FS.
        // The dimension is simply the number of variables needed to address all the space of the
        // input vector.
        let out_claims = trace
            .outputs()?
            .into_iter()
            .map(|out| {
                let r_i = self
                    .transcript
                    .read_challenges(out.get_data().len().ilog2() as usize);
                let y_i = out.get_data().to_vec().into_mle().evaluate(&r_i);
                Claim {
                    point: r_i,
                    eval: y_i,
                }
            })
            .collect_vec();

        let mut claims_by_layer: HashMap<NodeId, Vec<Claim<E>>> = HashMap::new();
        for (node_id, ctx) in self.ctx.steps_info.to_backward_iterator() {
            let InferenceStep {
                op: node_operation,
                step_data,
            } = trace
                .get_step(&node_id)
                .ok_or(anyhow!("Step in trace not found for node {}", node_id))?;
            trace!(
                "Proving node with id {node_id}: {:?}",
                node_operation.describe()
            );
            let claims_for_prove = ctx.claims_for_node(&claims_by_layer, &out_claims)?;
            let claims = if node_operation.is_provable() {
                node_operation.prove(node_id, &ctx.ctx, claims_for_prove, step_data, &mut self)?
            } else {
                // we only propagate the claims, without changing them, as a non-provable layer
                // shouldn't change the input values
                claims_for_prove.into_iter().cloned().collect()
            };
            claims_by_layer.insert(node_id, claims);
        }
        let span = metrics.to_span();
        stream_metrics("Claims", &span);
        debug!("== Claims generation metrics {} ==", span);

        // let trace_size = trace.last_step().id;

        // Now we have to make the table proofs
        debug!("== Generate proof ==");
        let metrics = Metrics::new();
        self.prove_tables()?;

        // now provide opening proofs for all claims accumulated during the proving steps
        let commit_proof = self
            .commit_prover
            .prove(&self.ctx.commitment_ctx, self.transcript)?;
        let output_proof = Proof {
            steps: self.proofs,
            table_proofs: self.table_proofs,
            commit: commit_proof,
        };

        let span = metrics.to_span();
        stream_metrics("Proof", &span);
        debug!("== Generate proof metrics {} ==", span);

        Ok(output_proof)
    }

    /// Looks at all the individual polys to accumulate from the witnesses and create the context from that.
    #[timed_instrument]
    fn instantiate_witness_ctx<'b>(
        &mut self,
        trace: &InferenceTrace<'b, E, Element>,
    ) -> anyhow::Result<()> {
        let LookupWitness {
            challenge_storage,
            logup_witnesses,
            table_witnesses,
        } = generate_lookup_witnesses::<E, T, PCS>(trace, self.ctx, self.transcript)?;
        self.challenge_storage = challenge_storage;
        self.lookup_witness = logup_witnesses;
        self.table_witness = table_witnesses;

        Ok(())
    }

    /// Prove only a subset of layers [start_layer, end_layer) using claims sourced from a prior proof state.
    /// This method mimics the proving logic from the main prove() method (lines 462-483).
    /// 
    /// # Arguments
    /// * `trace` - The inference trace containing execution data for all layers
    /// * `claims_by_layer` - Claims from previously proven layers, used as input for current layers
    /// * `out_claims` - Output claims for verification
    /// * `start_layer` - Starting layer index (inclusive)
    /// * `end_layer` - Ending layer index (exclusive)
    /// 
    /// # Returns
    /// * `HashMap<NodeId, Vec<Claim<E>>>` - Claims generated for the specified layer range
    pub fn prove_layers_from_trace<'b>(
        &mut self,
        trace: &InferenceTrace<'b, E, Element>,
        claims_by_layer: &HashMap<NodeId, Vec<Claim<E>>>,
        out_claims: &[Claim<E>],
        start_layer: usize,
        end_layer: usize,
    ) -> anyhow::Result<HashMap<NodeId, Vec<Claim<E>>>> {
        let forward_iter: Vec<_> = self.ctx.steps_info.to_forward_iterator().collect();

        ensure!(
            start_layer < end_layer && end_layer <= forward_iter.len(),
            "Invalid layer range: start={}, end={}, total_layers={}",
            start_layer,
            end_layer,
            forward_iter.len()
        );

        debug!("== Proving layers {} to {} ==", start_layer, end_layer - 1);
        let metrics = Metrics::new();
        let trace_fields = trace.clone().into_fields();

        // Build a fresh claims map for this partial proving - mimicking the main prove() method
        let mut new_claims_by_layer: HashMap<NodeId, Vec<Claim<E>>> = HashMap::new();

        // Prove only the selected range - mimicking the main prove() method logic (lines 462-483)
        for (node_id, ctx) in forward_iter[start_layer..end_layer].iter() {
            let InferenceStep {
                op: node_operation,
                step_data,
            } = trace_fields
                .get_step(node_id)
                .ok_or(anyhow!("Step in trace not found for node {}", node_id))?;

            trace!(
                "Proving node with id {node_id}: {:?}",
                node_operation.describe()
            );

            // Mimic the exact logic from main prove() method (line 474)
            let claims_for_prove = ctx.claims_for_node(claims_by_layer, out_claims)?;
            
            // Mimic the exact logic from main prove() method (lines 475-481)
            let claims = if node_operation.is_provable() {
                node_operation.prove(*node_id, &ctx.ctx, claims_for_prove, step_data, self)?
            } else {
                // we only propagate the claims, without changing them, as a non-provable layer
                // shouldn't change the input values
                claims_for_prove.into_iter().cloned().collect()
            };
            
            // Mimic the exact logic from main prove() method (line 482)
            new_claims_by_layer.insert(*node_id, claims);
        }

        let span = metrics.to_span();
        stream_metrics(&format!("Layers {}-{}", start_layer, end_layer - 1), &span);
        debug!("== Layers {} to {} completed in {} ==", start_layer, end_layer - 1, span);

        Ok(new_claims_by_layer)
    }
}
