import torch
import numpy as np

import pickle
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import gc

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

class PromptDataset(Dataset):
    def __init__(self, structured_prompts):
        self.prompts = [item['prompt'] for item in structured_prompts]
        self.prompt_types = [item['type'] for item in structured_prompts]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.prompt_types[idx]

class MultiGPUModelAnalyzer:
    def __init__(self, models_to_load, layer=-1, use_multi_gpu=True, batch_size=1):
        self.layer = layer
        self.use_multi_gpu = use_multi_gpu
        self.batch_size = batch_size
        
        if torch.cuda.is_available() and self.use_multi_gpu:
            self.gpu_count = torch.cuda.device_count()
            self.device_list = [f"cuda:{i}" for i in range(self.gpu_count)]
        else:
            self.gpu_count = 1
            self.device_list = ["cuda" if torch.cuda.is_available() else "cpu"]
        self.device = self.device_list[0]
        
        first_model_path = list(models_to_load.values())[0]
        self.tokenizer = AutoTokenizer.from_pretrained(first_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.models_to_load_paths = models_to_load
        self.hidden_states = {}
        self.prompt_types = None
        self.pca_results = None
        self.pca = None
        self.scaler = None
        self.classifier = None
        self.pca_data = None

    # ---------------- Extract Hidden States ----------------
    def get_hidden_states_for_model(self, model, dataloader, device):
        hidden_states_list = []
        model.eval()
        with torch.no_grad():
            for prompts_batch, _ in tqdm(dataloader, desc=f"Extracting hidden states (GPU: {device})"):
                formatted_texts = []
                for prompt in prompts_batch:
                    messages = [{"role": "user", "content": prompt}]
                    text = self.tokenizer.apply_chat_template(messages,
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    formatted_texts.append(text)
                inputs = self.tokenizer(formatted_texts, return_tensors="pt", padding=True,
                                        truncation=True, max_length=512).to(device)
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[self.layer][:, -1, :].squeeze()
                hidden_cpu = hidden.cpu().numpy()
                hidden_states_list.append(hidden_cpu)
                del outputs, hidden, inputs
                torch.cuda.empty_cache()
        if not hidden_states_list:
            return np.array([])
        return np.array(hidden_states_list) if hidden_states_list[0].ndim==1 else np.vstack(hidden_states_list)

    def process_prompts_multi_gpu(self, structured_prompts):
        dataset = PromptDataset(structured_prompts)
        self.prompt_types = dataset.prompt_types
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        for i, (name, path) in enumerate(self.models_to_load_paths.items()):
            device_idx = i % self.gpu_count
            current_device = self.device_list[device_idx]
            gc.collect(); torch.cuda.empty_cache()
            model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(current_device)
            self.hidden_states[name] = self.get_hidden_states_for_model(model, dataloader, current_device)
            del model; gc.collect(); torch.cuda.empty_cache()

    def process_prompts_single_gpu_sequential(self, structured_prompts):
        dataset = PromptDataset(structured_prompts)
        self.prompt_types = dataset.prompt_types
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        for name, path in self.models_to_load_paths.items():
            gc.collect(); torch.cuda.empty_cache()
            model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(self.device)
            self.hidden_states[name] = self.get_hidden_states_for_model(model, dataloader, self.device)
            del model; gc.collect(); torch.cuda.empty_cache()

    def process_prompts(self, structured_prompts):
        if self.use_multi_gpu and self.gpu_count>1:
            self.process_prompts_multi_gpu(structured_prompts)
        else:
            self.process_prompts_single_gpu_sequential(structured_prompts)

    # ---------------- PCA ----------------
    def perform_pca(self, n_components=2):
        all_hidden_list = [self.hidden_states[name] for name in sorted(self.models_to_load_paths.keys())
                           if self.hidden_states[name].size>0]
        if not all_hidden_list:
            self.pca_results = np.array([]); return
        all_hidden_stacked = np.vstack(all_hidden_list)
        self.scaler = StandardScaler()
        all_hidden_scaled = self.scaler.fit_transform(all_hidden_stacked)
        self.pca = PCA(n_components=n_components)
        self.pca_results = self.pca.fit_transform(all_hidden_scaled)
        print(f"PCA Explained Variance Ratio: {self.pca.explained_variance_ratio_}")

    # ---------------- Data Preparation & Classifier ----------------
    def _prepare_data_and_classifier(self, base_model_name):
        self.pca_data = {}
        current_offset=0
        for name in sorted(self.models_to_load_paths.keys()):
            if self.hidden_states[name].size>0:
                num_samples = self.hidden_states[name].shape[0]
                self.pca_data[name] = self.pca_results[current_offset:current_offset+num_samples]
                current_offset+=num_samples
        base_pca = self.pca_data.get(base_model_name)
        if base_pca is None or base_pca.size==0:
            return None, None
        focus_types = {'harmful':1, 'infer':0}
        X_train,y_train=[],[]
        for idx, pt in enumerate(self.prompt_types):
            if pt in focus_types:
                X_train.append(base_pca[idx])
                y_train.append(focus_types[pt])
        X_train, y_train = np.array(X_train), np.array(y_train)
        self.classifier=None
        if len(np.unique(y_train))>1:
            self.classifier = LogisticRegression(solver='lbfgs')
            self.classifier.fit(X_train,y_train)
        return self.pca_data, self.classifier

    # ---------------- Visualization ----------------
    def _plot_confidence_ellipse(self, ax, points, n_std=2.0, **kwargs):
        if points.shape[0]<3: return
        centroid = np.mean(points, axis=0)
        cov = np.cov(points, rowvar=False)
        eigvals,eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals,eigvecs = eigvals[order],eigvecs[:,order]
        angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
        width,height=2*n_std*np.sqrt(eigvals)
        ellipse=Ellipse(xy=centroid,width=width,height=height,angle=angle,**kwargs)
        ax.add_patch(ellipse)
            
    # ==================== Visualization of Centroid Shifts ====================
    def visualize_centroid_comparison(self, base_model_name, tuned_model_names, save_path,
                                use_subplots=False, title=None,
                                save_centroid_data=False, load_centroid_json=None):
        """
        Visualize centroid shifts, decision boundary, and confidence ellipses (focus on harmful and reasoning prompts),
        with directional movement distances displayed in the top-right corner.

        Args:
            base_model_name: Name of the base model
            tuned_model_names: List of tuned model names
            save_path: Path to save the figure
            use_subplots: Whether to use subplot layout
            title: Figure title
            save_centroid_data: Whether to save centroid data to JSON
            load_centroid_json: Path to load centroid data from JSON
        """
        import json, os
        from matplotlib.patches import FancyArrowPatch

        plt.switch_backend('Agg')

        # Load centroid data from JSON, skipping PCA-dependent calculations
        centroid_data = None
        loaded_mode = False
        if load_centroid_json:
            with open(load_centroid_json, 'r', encoding='utf-8') as f:
                centroid_data = json.load(f)
            loaded_mode = True

        # Standard workflow (if not loading from JSON)
        if not loaded_mode:
            if self.pca_results is None or self.pca is None:
                raise ValueError("Please run perform_pca first")
            if self.prompt_types is None:
                raise ValueError("prompt_types not set")
            if self.pca_data is None or self.classifier is None:
                self.pca_data, self.classifier = self._prepare_data_and_classifier(base_model_name)
            if self.pca_data is None:
                return

        # Determine subplot layout based on number of tuned models
        num_tuned = len(tuned_model_names)
        if use_subplots and num_tuned > 1:
            fig, axes = plt.subplots(1, num_tuned, figsize=(12 * num_tuned, 10), sharex=True, sharey=True)
        else:
            fig, ax = plt.subplots(figsize=(14, 10))
            axes = [ax] * num_tuned

        model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        model_color_map = {name: model_colors[i % len(model_colors)] for i, name in enumerate(tuned_model_names)}

        focus_types = ['harmful', 'infer']
        prompt_style_map = {
            'harmful': {'edgecolor': 'red', 'marker': 'X', 'label': 'Harmful'},
            'infer': {'edgecolor': 'blue', 'marker': 'P', 'label': 'Reasoning'}
        }

        # Collect centroid data if not loaded from JSON
        collected_centroids = [] if not loaded_mode else centroid_data

        if not loaded_mode:
            base_pca = self.pca_data[base_model_name]

            for i, model_name in enumerate(tuned_model_names):
                ax = axes[i] if use_subplots and num_tuned > 1 else axes[0]
                tuned_pca = self.pca_data.get(model_name)
                if tuned_pca is None or tuned_pca.size == 0:
                    continue

                # --- Move title to bottom ---
                if title:
                    title_text = f"{title}: {base_model_name} -> {model_name}" if use_subplots and num_tuned > 1 else title
                    ax.text(0.5, -0.2, title_text, transform=ax.transAxes, ha='center', va='center', fontsize=26, weight='bold')

                # Collect delta info for top-right display
                delta_info = []

                for p_type in focus_types:
                    idxs = [k for k, v in enumerate(self.prompt_types) if v == p_type]
                    if not idxs:
                        continue
                    style = prompt_style_map[p_type]

                    # Draw confidence ellipses
                    self._plot_confidence_ellipse(ax, base_pca[idxs], facecolor='gray', edgecolor='gray',
                                                alpha=0.3, linestyle=':', linewidth=3, zorder=3)
                    self._plot_confidence_ellipse(ax, tuned_pca[idxs], facecolor=style['edgecolor'], edgecolor=style['edgecolor'],
                                                alpha=0.25, linewidth=3.5, zorder=4)

                    # Compute centroids
                    base_c = np.mean(base_pca[idxs], axis=0)
                    tuned_c = np.mean(tuned_pca[idxs], axis=0)
                    dx, dy = tuned_c[0] - base_c[0], tuned_c[1] - base_c[1]

                    # --- Compute directional distance ---
                    if self.classifier is not None:
                        w = self.classifier.coef_[0]
                        b = self.classifier.intercept_[0]
                        norm = np.sqrt(np.sum(w**2))
                        base_safety_dist = np.abs(w.dot(base_c) + b) / norm
                        tuned_safety_dist = np.abs(w.dot(tuned_c) + b) / norm
                        distance = tuned_safety_dist - base_safety_dist
                    else:
                        distance = np.sqrt(dx ** 2 + dy ** 2)
                    # --- End of directional distance ---

                    # Draw centroid arrows
                    ax.arrow(base_c[0], base_c[1], dx, dy, color=style['edgecolor'], alpha=0.9,
                            width=0.04, head_width=0.25, length_includes_head=True, zorder=10)

                    # Collect delta info
                    delta_info.append({
                        'type': style['label'],
                        'distance': distance,
                        'color': style['edgecolor']
                    })

                    # Draw centroid points
                    ax.scatter(base_c[0], base_c[1], c='black', marker=style['marker'], s=180,
                            edgecolors='white', linewidth=1.5, zorder=6)
                    ax.scatter(tuned_c[0], tuned_c[1], c=style['edgecolor'], marker=style['marker'], s=180,
                            edgecolors='white', linewidth=1.5, zorder=6)

                    # Save centroid data
                    collected_centroids.append({
                        "prompt_type": p_type,
                        "base_model": base_model_name,
                        "tuned_model": model_name,
                        "base_c": base_c.tolist(),
                        "tuned_c": tuned_c.tolist(),
                        "delta": [float(dx), float(dy)],
                        "distance": float(distance),
                        "euclidean_distance": float(np.sqrt(dx**2 + dy**2))
                    })

                # Display delta in top-right corner
                delta_text = ""
                for info in delta_info:
                    if self.classifier is not None:
                        delta_text += f"Δ{info['type']}: {info['distance']:+.3f}\n"
                    else:
                        delta_text += f"Δ{info['type']}: {info['distance']:.3f}\n"

                if delta_text:
                    ax.text(0.98, 0.98, delta_text.strip(), transform=ax.transAxes,
                            fontsize=18, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1),
                            zorder=15)

                # Draw decision boundary
                if self.classifier:
                    x_min, x_max = ax.get_xlim()
                    y_min, y_max = ax.get_ylim()
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
                    Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2, zorder=5)

        # Load mode: plot centroids/arrows/labels from JSON
        else:
            grouped = {name: [] for name in tuned_model_names}
            for item in (collected_centroids or []):
                tm = item.get("tuned_model")
                if tm in grouped:
                    grouped[tm].append(item)

            for i, model_name in enumerate(tuned_model_names):
                ax = axes[i] if use_subplots and num_tuned > 1 else axes[0]
                items = grouped.get(model_name, [])

                # Move title to bottom
                if title:
                    title_text = f"{title}: {base_model_name} -> {model_name}" if use_subplots and num_tuned > 1 else title
                    ax.text(0.5, -0.2, title_text, transform=ax.transAxes, ha='center', va='center', fontsize=26, weight='bold')

                delta_info = []

                for item in items:
                    p_type = item["prompt_type"]
                    style = prompt_style_map.get(p_type, {'edgecolor': 'black', 'marker': 'o', 'label': p_type})
                    base_c = np.array(item["base_c"])
                    tuned_c = np.array(item["tuned_c"])
                    dx, dy = item["delta"]
                    distance = item["distance"]

                    # Arrows & points (no ellipses or decision boundary)
                    ax.arrow(base_c[0], base_c[1], dx, dy, color=style['edgecolor'], alpha=0.9,
                            width=0.04, head_width=0.25, length_includes_head=True, zorder=10)
                    ax.scatter(base_c[0], base_c[1], c='black', marker=style['marker'], s=180,
                            edgecolors='white', linewidth=1.5, zorder=6)
                    ax.scatter(tuned_c[0], tuned_c[1], c=style['edgecolor'], marker=style['marker'], s=180,
                            edgecolors='white', linewidth=1.5, zorder=6)

                    delta_info.append({
                        'type': style['label'],
                        'distance': distance,
                        'color': style['edgecolor']
                    })

                # Display delta info
                delta_text = ""
                for info in delta_info:
                    direction = "Away" if info['distance'] >= 0 else "Closer"
                    delta_text += f"Δ{info['type']}: {info['distance']:.3f} ({direction})\n"

                if delta_text:
                    ax.text(0.98, 0.98, delta_text.strip(), transform=ax.transAxes,
                            fontsize=18, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1),
                            zorder=15)

        # ---------------- Custom legend ----------------
        from matplotlib.patches import Ellipse
        from matplotlib.lines import Line2D

        legend_elements = []

        # Decision boundary
        if (not loaded_mode) and self.classifier:
            legend_elements.append(Line2D([0], [0], color='black', linestyle='--', lw=2, label='Decision Boundary'))

        # Clusters
        if not loaded_mode:
            legend_elements.append((Ellipse((0,0),0.3,0.15,facecolor='gray',edgecolor='gray',alpha=0.6,linestyle=':'),'Base Model Cluster'))
            legend_elements.append((Ellipse((0,0),0.3,0.15,facecolor='red',edgecolor='red',alpha=0.6),'Harmful Cluster'))
            legend_elements.append((Ellipse((0,0),0.3,0.15,facecolor='blue',edgecolor='blue',alpha=0.6),'Reasoning Cluster'))

        # Centroids
        legend_elements.append(Line2D([0],[0],marker='X',color='w',label='Harmful Centroid (Base)',markerfacecolor='black',markeredgecolor='white',markersize=10,linewidth=0))
        legend_elements.append(Line2D([0],[0],marker='P',color='w',label='Reasoning Centroid (Base)',markerfacecolor='black',markeredgecolor='white',markersize=10,linewidth=0))
        legend_elements.append(Line2D([0],[0],marker='X',color='w',label='Harmful Centroid (Tuned)',markerfacecolor='red',markeredgecolor='white',markersize=10,linewidth=0))
        legend_elements.append(Line2D([0],[0],marker='P',color='w',label='Reasoning Centroid (Tuned)',markerfacecolor='blue',markeredgecolor='white',markersize=10,linewidth=0))

        handles, labels = [], []
        for item in legend_elements:
            if isinstance(item, tuple):
                handles.append(item[0])
                labels.append(item[1])
            else:
                handles.append(item)
                labels.append(item.get_label())

        fig.legend(handles=handles, labels=labels, fontsize=24,
                bbox_to_anchor=(0,0.95,1,0.1), loc='lower left', ncol=4,
                mode='expand', borderaxespad=0., frameon=False)

        for ax in axes:
            if self.pca is not None:
                ax.set_xlabel(f'PC 1 ({self.pca.explained_variance_ratio_[0]:.2%})', fontsize=18)
                ax.set_ylabel(f'PC 2 ({self.pca.explained_variance_ratio_[1]:.2%})', fontsize=18)
            else:
                ax.set_xlabel('PC 1', fontsize=14)
                ax.set_ylabel('PC 2', fontsize=14)
            ax.grid(True, alpha=0.3)

        plt.subplots_adjust(top=0.85, bottom=0.2)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)

        if save_centroid_data and (collected_centroids is not None):
            base, ext = os.path.splitext(save_path)
            json_path = base + "_centroid_data.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(collected_centroids, f, indent=2, ensure_ascii=False)
            print(f"[+] Centroid data saved to {json_path}")

        print(f"[+] Visualization saved to: {save_path}")

    
    # ---------------- Save and Load ----------------
    def save_all_data(self, save_path):
        """
        Save all computed results to a file to avoid recomputation
        """
        data_to_save = {
            'hidden_states': self.hidden_states,
            'prompt_types': self.prompt_types,
            'pca_results': self.pca_results,
            'pca_model': self.pca,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'pca_data': self.pca_data,
            'models_to_load_paths': self.models_to_load_paths,
            'layer': self.layer
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"[+] All data saved to: {save_path}")
    
    def load_all_data(self, load_path):
        """
        Load all previously computed results, skipping model loading and computation
        """
        if not os.path.exists(load_path):
            print(f"[-] Data file not found: {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            self.hidden_states = loaded_data['hidden_states']
            self.prompt_types = loaded_data['prompt_types']
            self.pca_results = loaded_data['pca_results']
            self.pca = loaded_data['pca_model']
            self.scaler = loaded_data['scaler']
            self.classifier = loaded_data['classifier']
            self.pca_data = loaded_data['pca_data']
            self.models_to_load_paths = loaded_data['models_to_load_paths']
            self.layer = loaded_data['layer']
            
            print(f"[+] Successfully loaded data: {load_path}")
            print(f"    - Models included: {list(self.hidden_states.keys())}")
            print(f"    - Number of prompts: {len(self.prompt_types)}")
            print(f"    - PCA dimensions: {self.pca_results.shape if self.pca_results is not None else 'None'}")
            
            return True
            
        except Exception as e:
            print(f"[-] Failed to load data: {e}")
            return False
    
    def get_data_summary(self):
        """
        Return a summary of current data status
        """
        summary = {
            'has_hidden_states': len(self.hidden_states) > 0,
            'has_pca_results': self.pca_results is not None,
            'has_classifier': self.classifier is not None,
            'models': list(self.hidden_states.keys()) if self.hidden_states else [],
            'num_prompts': len(self.prompt_types) if self.prompt_types else 0
        }
        return summary

    # ---------------- Quantitative Analysis ----------------
    # ---------------- Quantitative Analysis ----------------
    def run_quantitative_analysis(self, base_model_name, output_dir):
        if self.pca_data is None or self.classifier is None:
            self.pca_data, self.classifier = self._prepare_data_and_classifier(base_model_name)
        if self.pca_data is None: 
            return

        results = []
        focus_types = {'harmful': 1, 'infer': 0}

        # ---------------- Safety Distance ----------------
        if self.classifier:
            w = self.classifier.coef_[0]
            b = self.classifier.intercept_[0]
            norm = np.sqrt(np.sum(w ** 2))
            for model_name, points in self.pca_data.items():
                for p_type in focus_types:
                    idxs = [i for i, pt in enumerate(self.prompt_types) if pt == p_type]
                    if not idxs: 
                        continue
                    centroid = np.mean(points[idxs], axis=0)
                    dist = np.abs(w.dot(centroid) + b) / norm
                    results.append({
                        'model': model_name,
                        'metric': 'Safety Distance',
                        'prompt_type': p_type,
                        'value': dist
                    })

        # ---------------- Save Results ----------------
        df = pd.DataFrame(results)
        pivot_df = df.pivot_table(index=['model', 'metric'], columns='prompt_type', values='value').reset_index()
        os.makedirs(output_dir, exist_ok=True)
        excel_path = os.path.join(output_dir, 'quantitative_analysis_results.xlsx')
        pivot_df.to_excel(excel_path, index=False)


def main():
    # --- 1. Configuration ---
    models_to_load = {
        'Base': 'Qwen/Qwen3-32B',
        'SafR': 'Qwen3_32B_SafR',
        'SafB': 'Qwen3_32B_SafB',
        'XXX': 'Qwen3_32B_XXX'
    }
    input_file_path = "./Data/PCA_data.jsonl"
    output_dir = "./Result/pca_output/example"
    
    # Full data save path
    full_data_path = os.path.join(output_dir, "full_analysis_data.pkl")
    
    layer_to_analyze = -1
    
    # --- 2. Initialize Analyzer ---
    analyzer = MultiGPUModelAnalyzer(models_to_load, layer_to_analyze, use_multi_gpu=True, batch_size=4)
    
    # --- 3. Attempt to Load Existing Data ---
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(full_data_path):
        print("--- Existing data file found, attempting to load ---")
        if analyzer.load_all_data(full_data_path):
            print("--- Successfully loaded existing data, skipping model loading ---")
        else:
            print("--- Failed to load data, performing full computation ---")
            analyzer = None  # Reinitialize
    
    # --- 4. If loading failed, perform full computation ---
    if analyzer is None or not analyzer.get_data_summary()['has_hidden_states']:
        print("--- Performing full model loading and computation ---")
        analyzer = MultiGPUModelAnalyzer(models_to_load, layer_to_analyze, use_multi_gpu=True, batch_size=4)
        
        # Load prompts
        print(f"--- Loading prompts from {input_file_path} ---")
        structured_prompts = []
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        structured_prompts.append({'prompt': item['content'], 'type': item['type']})
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Skipping malformed or missing key line: {line.strip()} | Error: {e}")
        except FileNotFoundError:
            print(f"Error: Input file not found! Please check path: {input_file_path}")
            return 
            
        if not structured_prompts:
            print("Error: No valid prompts loaded from file. Terminating program.")
            return
            
        print(f"Successfully loaded {len(structured_prompts)} prompts.")
        
        # Model processing and PCA computation
        analyzer.process_prompts(structured_prompts)
        analyzer.perform_pca()
        
        # Save all computed results
        analyzer.save_all_data(full_data_path)
    
    # --- 5. Generate Visualization ---
    print("\n--- Generating visualization plots ---")
    
    # Compare all models
    analyzer.visualize_centroid_comparison(
        base_model_name='Base',
        tuned_model_names=['SafR', 'SafB', 'SafeChain'],
        save_path=f"{output_dir}/centroid_comparison_all.pdf",
        use_subplots=True,
        title="Centroid Shift Analysis"
    )
    
    # --- 6. Quantitative Analysis ---
    print("\n--- Running quantitative analysis ---")
    analyzer.run_quantitative_analysis('Base', output_dir)
    
    print("\n--- All analyses completed! ---")
    print(f"Data saved to: {full_data_path}")
    print("Next run will load existing data directly without recomputation.")


if __name__ == "__main__":
    main()
