##########################################################################
# File Name: rapidock_inference.py
# Author: AI Assistant
# Created Time: 2024
# Description: RAPiDock inference class for API usage
#########################################################################

import os
import copy
import yaml
import torch
import MDAnalysis
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from io import StringIO
from argparse import Namespace
from MDAnalysis.coordinates.memory import MemoryReader
from torch_geometric.loader import DataListLoader
from RAPiDock.utils.utils import get_model, ExponentialMovingAverage
from RAPiDock.utils.inference_utils import InferenceDataset, set_nones
from RAPiDock.utils.peptide_updater import randomize_position
from RAPiDock.utils.sampling import sampling
import multiprocessing

warnings.filterwarnings("ignore")


class RAPiDockInference:
    """
    RAPiDock inference class for API usage
    """

    def __init__(
        self,
        model_dir,
        ckpt="best_model.pt",
        confidence_model_dir=None,
        confidence_ckpt="best_model.pt",
        device=None,
    ):
        """
        Initialize RAPiDock inference

        Args:
            model_dir: Path to model directory
            ckpt: Model checkpoint filename
            confidence_model_dir: Path to confidence model directory (optional)
            confidence_ckpt: Confidence model checkpoint filename
            device: torch device (if None, auto-detect)
        """
        self.model_dir = model_dir
        self.ckpt = ckpt
        self.confidence_model_dir = confidence_model_dir
        self.confidence_ckpt = confidence_ckpt
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load model parameters
        with open(f"{self.model_dir}/model_parameters.yml") as f:
            self.score_model_args = Namespace(**yaml.full_load(f))

        # Load models
        self.model = self._load_model()
        self.confidence_model = None
        self.confidence_args = None

    def _load_model(self):
        """Load the main score model"""
        model = get_model(self.score_model_args, no_parallel=True)
        state_dict = torch.load(
            f"{self.model_dir}/{self.ckpt}", map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict["model"], strict=True)
        model = model.to(self.device)

        ema_weights = ExponentialMovingAverage(
            model.parameters(), decay=self.score_model_args.ema_rate
        )
        ema_weights.load_state_dict(state_dict["ema_weights"], device=self.device)
        ema_weights.copy_to(model.parameters())
        return model

    def _load_confidence_model(self):
        """Load confidence model if specified"""
        if not self.confidence_model_dir:
            return None

        with open(f"{self.confidence_model_dir}/model_parameters.yml") as f:
            self.confidence_args = Namespace(**yaml.full_load(f))

        confidence_model = get_model(
            self.confidence_args, no_parallel=True, confidence_mode=True
        )
        state_dict = torch.load(
            f"{self.confidence_model_dir}/{self.confidence_ckpt}",
            map_location=torch.device("cpu"),
        )
        confidence_model.load_state_dict(state_dict["model"], strict=True)
        confidence_model = confidence_model.to(self.device)
        confidence_model.eval()
        return confidence_model

    def _prepare_data(
        self,
        output_dir,
        complex_name_list,
        protein_description_list,
        peptide_description_list,
        conformation_type="random",
        conformation_partial=None,
    ):
        """Prepare inference dataset"""
        # Create output directories
        for name in complex_name_list:
            write_dir = f"{output_dir}/{name}"
            os.makedirs(write_dir, exist_ok=True)

        return InferenceDataset(
            output_dir=output_dir,
            complex_name_list=complex_name_list,
            protein_description_list=protein_description_list,
            peptide_description_list=peptide_description_list,
            lm_embeddings=self.score_model_args.esm_embeddings_path_train is not None,
            lm_embeddings_pep=self.score_model_args.esm_embeddings_peptide_train
            is not None,
            conformation_type=conformation_type,
            conformation_partial=conformation_partial,
        )

    def _prepare_data_list(self, original_complex_graph, N):
        """Prepare data list for inference"""
        data_list = []
        nums = []
        if len(original_complex_graph["peptide_inits"]) == 1:
            data_list = [copy.deepcopy(original_complex_graph) for _ in range(N)]
        elif len(original_complex_graph["peptide_inits"]) > 1:
            for i, peptide_init in enumerate(original_complex_graph["peptide_inits"]):
                if i != 0:
                    original_complex_graph["pep_a"].pos = (
                        torch.from_numpy(
                            MDAnalysis.Universe(peptide_init).atoms.positions
                        )
                        - original_complex_graph.original_center
                    )
                num = (
                    N - sum(nums)
                    if i == len(original_complex_graph["peptide_inits"]) - 1
                    else round(
                        original_complex_graph["partials"][i]
                        / sum(original_complex_graph["partials"])
                        * N
                    )
                )
                nums.append(num)
                data_list.extend(
                    [copy.deepcopy(original_complex_graph) for _ in range(num)]
                )
        return data_list

    def _save_predictions(
        self,
        write_dir,
        predict_pos,
        original_complex_graph,
        confidence=None,
        scoring_function="confidence",
        fastrelax=False,
        cpu=1,
    ):
        """Save prediction results"""
        raw_pdb = MDAnalysis.Universe(
            StringIO(original_complex_graph["pep"].noh_mda), format="pdb"
        )
        peptide_unrelaxed_files = []

        re_order = None
        # reorder predictions based on confidence output
        if confidence is not None:
            confidence = confidence.cpu().numpy()
            re_order = np.argsort(confidence)[::-1]
            confidence = confidence[re_order]
            predict_pos = predict_pos[re_order]

        for rank, pos in enumerate(predict_pos):
            raw_pdb.atoms.positions = pos
            file_name = (
                f"rank{rank+1}_{scoring_function}.pdb"
                if confidence is not None
                else f"rank{rank+1}.pdb"
            )
            peptide_unrelaxed_file = os.path.join(write_dir, file_name)
            peptide_unrelaxed_files.append(peptide_unrelaxed_file)
            raw_pdb.atoms.write(peptide_unrelaxed_file)

        if scoring_function == "ref2015" or fastrelax:
            from RAPiDock.utils.pyrosetta_utils import relax_score

            relaxed_poses = [
                peptide.replace(".pdb", "_relaxed.pdb")
                for peptide in peptide_unrelaxed_files
            ]
            protein_raw_file = (
                f"{write_dir}/{os.path.basename(write_dir)}_protein_raw.pdb"
            )

            with multiprocessing.Pool(cpu) as pool:
                ref2015_scores = pool.map(
                    relax_score,
                    zip(
                        [protein_raw_file] * len(peptide_unrelaxed_files),
                        peptide_unrelaxed_files,
                        relaxed_poses,
                        [scoring_function == "ref2015"] * len(peptide_unrelaxed_files),
                    ),
                )
            if ref2015_scores and ref2015_scores[0] is not None:
                re_order = np.argsort(ref2015_scores)
                score_results = [["file", "ref2015score"]]
                for rank, order in enumerate(re_order):
                    os.rename(
                        relaxed_poses[order],
                        os.path.join(write_dir, f"rank{rank+1}_{scoring_function}.pdb"),
                    )
                    score_results.append(
                        [
                            f"rank{rank+1}_{scoring_function}",
                            f"{ref2015_scores[order]:.2f}",
                        ]
                    )
                open(os.path.join(write_dir, "ref2015_score.csv"), "w").write(
                    "\n".join([",".join(i) for i in score_results])
                )

        return re_order if re_order is not None else 0

    def _save_visualization(
        self,
        write_dir,
        visualization_list,
        original_complex_graph,
        predict_pos,
        scoring_function="confidence",
        re_order=None,
    ):
        """Save visualization frames"""
        raw_pdb = MDAnalysis.Universe(
            StringIO(original_complex_graph["pep"].noh_mda), format="pdb"
        )
        visualization_list = list(
            np.transpose(np.array(visualization_list), (1, 0, 2, 3))
        )
        if scoring_function in ["confidence", "ref2015"] and re_order is not None:
            for rank, batch_idx in enumerate(re_order):
                raw_pdb.load_new(visualization_list[batch_idx], format=MemoryReader)
                with MDAnalysis.Writer(
                    os.path.join(write_dir, f"rank{rank+1}_reverseprocess.pdb"),
                    multiframe=True,
                    bonds=None,
                    n_atoms=raw_pdb.atoms.n_atoms,
                ) as pdb_writer:
                    for ts in raw_pdb.trajectory:
                        pdb_writer.write(raw_pdb)
        else:
            for rank in range(len(predict_pos)):
                raw_pdb.load_new(visualization_list[rank], format=MemoryReader)
                with MDAnalysis.Writer(
                    os.path.join(write_dir, f"rank{rank+1}_reverseprocess.pdb"),
                    multiframe=True,
                    bonds=None,
                    n_atoms=raw_pdb.atoms.n_atoms,
                ) as pdb_writer:
                    for ts in raw_pdb.trajectory:
                        pdb_writer.write(raw_pdb)

    def process_complex(
        self,
        original_complex_graph,
        write_dir,
        N=20,
        batch_size=10,
        no_final_step_noise=False,
        inference_steps=20,
        actual_steps=None,
        save_visualisation=False,
        scoring_function="confidence",
        fastrelax=False,
        cpu=1,
    ):
        """Process a single complex"""
        # Load confidence model if needed
        if scoring_function == "confidence" and self.confidence_model is None:
            self.confidence_model = self._load_confidence_model()

        # Prepare data list
        data_list = self._prepare_data_list(original_complex_graph, N)
        randomize_position(data_list, False, self.score_model_args.tr_sigma_max)

        visualization_list = None
        if save_visualisation:
            visualization_list = [
                np.asarray(
                    [
                        g["pep_a"].pos.cpu().numpy()
                        + original_complex_graph.original_center.cpu().numpy()
                        for g in data_list
                    ]
                )
            ]

        data_list, confidence, visualization_list = sampling(
            data_list=data_list,
            model=self.model,
            args=self.score_model_args,
            batch_size=batch_size,
            no_final_step_noise=no_final_step_noise,
            inference_steps=inference_steps,
            actual_steps=actual_steps if actual_steps is not None else inference_steps,
            visualization_list=visualization_list,
            confidence_model=self.confidence_model,
        )

        predict_pos = np.asarray(
            [
                complex_graph["pep_a"].pos.cpu().numpy()
                + original_complex_graph.original_center.cpu().numpy()
                for complex_graph in data_list
            ]
        )

        # Save predictions
        re_order = self._save_predictions(
            write_dir,
            predict_pos,
            original_complex_graph,
            confidence,
            scoring_function,
            fastrelax,
            cpu,
        )

        # Save visualization frames
        if save_visualisation:
            self._save_visualization(
                write_dir,
                visualization_list,
                original_complex_graph,
                predict_pos,
                scoring_function,
                re_order,
            )

        return predict_pos, confidence, re_order

    def run_inference(
        self,
        output_dir,
        complex_name_list=None,
        protein_description_list=None,
        peptide_description_list=None,
        protein_peptide_csv=None,
        complex_name=None,
        protein_description=None,
        peptide_description=None,
        N=20,
        batch_size=10,
        no_final_step_noise=False,
        inference_steps=20,
        actual_steps=None,
        save_visualisation=False,
        scoring_function="ref2015",
        fastrelax=False,
        cpu=1,
        conformation_type="E",
        conformation_partial=None,
    ):
        """
        Run complete inference process

        Args:
            output_dir: Output directory path
            complex_name_list: List of complex names
            protein_description_list: List of protein descriptions (file paths or sequences)
            peptide_description_list: List of peptide descriptions (file paths or sequences)
            protein_peptide_csv: CSV file with complex information
            complex_name: Single complex name (alternative to lists)
            protein_description: Single protein description (alternative to lists)
            peptide_description: Single peptide description (alternative to lists)
            N: Number of samples to generate
            batch_size: Batch size for inference
            no_final_step_noise: Whether to add noise in final step
            inference_steps: Number of inference steps
            actual_steps: Actual steps (if different from inference_steps)
            save_visualisation: Whether to save visualization
            scoring_function: Scoring function ("confidence" or "ref2015")
            fastrelax: Whether to use fast relax
            cpu: Number of CPU cores for multiprocessing
            conformation_type: Conformation type
            conformation_partial: Partial conformation

        Returns:
            dict: Results containing predictions for each complex
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare input data
        if protein_peptide_csv is not None:
            df = pd.read_csv(protein_peptide_csv)
            complex_name_list = set_nones(df["complex_name"].tolist())
            protein_description_list = set_nones(df["protein_description"].tolist())
            peptide_description_list = set_nones(df["peptide_description"].tolist())
        else:
            complex_name_list = complex_name_list or [complex_name]
            protein_description_list = protein_description_list or [protein_description]
            peptide_description_list = peptide_description_list or [peptide_description]

        complex_name_list = [
            name if name is not None else f"complex_{i}"
            for i, name in enumerate(complex_name_list)
        ]

        # Prepare dataset
        inference_dataset = self._prepare_data(
            output_dir,
            complex_name_list,
            protein_description_list,
            peptide_description_list,
            conformation_type,
            conformation_partial,
        )

        inference_loader = DataListLoader(
            dataset=inference_dataset, batch_size=1, shuffle=False
        )

        # Process complexes
        results = {}
        failures, skipped = 0, 0
        print("Size of test dataset: ", len(inference_dataset))

        for idx, original_complex_graph in tqdm(enumerate(inference_loader)):
            complex_name = inference_dataset.complex_names[idx]

            if not original_complex_graph[0].success:
                skipped += 1
                print(f"SKIPPING | Complex {complex_name} failed to process")
                continue

            try:
                write_dir = f"{output_dir}/{complex_name}"
                predict_pos, confidence, re_order = self.process_complex(
                    original_complex_graph[0],
                    write_dir,
                    N,
                    batch_size,
                    no_final_step_noise,
                    inference_steps,
                    actual_steps,
                    save_visualisation,
                    scoring_function,
                    fastrelax,
                    cpu,
                )

                results[complex_name] = {
                    "predictions": predict_pos,
                    "confidence": confidence,
                    "reorder": re_order,
                    "output_dir": write_dir,
                }

            except Exception as e:
                print(f"Failed on {complex_name}: {e}")
                failures += 1

        return results


if __name__ == "__main__":
    rapidock = RAPiDockInference(
        model_dir="/xcfhome/zpzeng/gitrepo/RAPiDock/train_models/CGTensorProductEquivariantModel/",
        ckpt="rapidock_global.pt",
    )

    results = rapidock.run_inference(
        output_dir="/xcfhome/zpzeng/gitrepo/RAPiDock/results/RefPepDB-RecentSet",
        complex_name="my_complex",
        protein_description="/xcfhome/zpzeng/gitrepo/RAPiDock/chain_A.pdb",
        peptide_description="/xcfhome/zpzeng/gitrepo/RAPiDock/chain_B.pdb",
        N=3,
        cpu=20
    )
