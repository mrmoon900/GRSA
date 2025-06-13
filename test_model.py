import torch
import logging
import os
import json
import pandas as pd
from data_preprocess import dataload, create_optimized_dataloaders
from molecular_model import MolecularNetwork
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, mean_squared_error, 
                           mean_absolute_error, r2_score, confusion_matrix,
                           roc_curve, auc, precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)

def fix_state_dict_keys(state_dict, use_contrastive=True):
 
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('smiles_processor.'):
            new_key = key.replace('smiles_processor.', 'features_processor.')
        if not use_contrastive and key.startswith('contrastive_learner.'):
            print(f"Skipping contrastive learning parameter: {key}")
            continue

        new_state_dict[new_key] = value
    return new_state_dict

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary),
        "recall": recall_score(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }
    return metrics

def get_dataset_mapping(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'dataset_mapping' in checkpoint:
            return checkpoint['dataset_mapping']
    except:
        logging.warning(f"Could not load dataset mapping from checkpoint: {checkpoint_path}")
    
    metadata_path = checkpoint_path.replace('.pth', '_metadata.json')
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(os.path.dirname(checkpoint_path), 'dataset_mapping.json')
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if 'dataset_mapping' in metadata:
                    return metadata['dataset_mapping']
        except:
            logging.warning(f"Could not load dataset mapping from metadata: {metadata_path}")
    return {
        'train': 'Main Test',
        'val': 'TS1',
        'test': 'TS2',
        'ts1': 'TS3'
    }

def save_results(results, all_predictions, all_labels, result_path):
    os.makedirs(result_path, exist_ok=True)

    metrics_data = []
    for dataset_name, result in results.items():
        metrics = result['metrics']
        loss = result['loss']

        metrics_data.append({
            'Dataset': dataset_name,
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-score': f"{metrics['f1']:.3f}",
            'ROC-AUC': f"{metrics['roc_auc']:.3f}",
            'PR-AUC': f"{metrics['pr_auc']:.3f}",  
            'Sensitivity': f"{metrics['sensitivity']:.3f}",
            'Specificity': f"{metrics['specificity']:.3f}",
            'R2': f"{metrics['r2']:.3f}",
            'Loss': f"{loss:.3f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    print("\nConfusion Matrices:")
    print("=" * 50)
    for dataset_name in results.keys():
        if dataset_name in all_predictions and dataset_name in all_labels:
            pred_labels = (np.array(all_predictions[dataset_name]) > 0.5).astype(int)
            true_labels = np.array(all_labels[dataset_name])
            cm = confusion_matrix(true_labels, pred_labels)
            print(f"\n{dataset_name} Confusion Matrix:")
            print(cm)
            print("-" * 50)

    print("\nMetrics Summary:")
    print("=" * 130)
    header = "Dataset    "
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC","PR-AUC", "Sensitivity", "Specificity", "R2", "Loss"]
    for metric in metrics:
        header += f"{metric:<12}"
    print(header)
    print("-" * 130)
    for _, row in metrics_df.iterrows():
        line = f"{row['Dataset']:<10}"
        for metric in metrics:
            line += f"{row[metric]:<12}"
        print(line)
    print("=" * 130)
    metrics_df.to_csv(os.path.join(result_path, 'metrics_summary.csv'), index=False)
    #Generate plots
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    name_mapping = {
        'test': 'Main Test',
        'ts1': 'TS1', 
        'ts2': 'TS2', 
        'ts3': 'TS3'
    }
    colors = {'Main Test': '#1f77b4', 'TS1': '#ff7f0e', 'TS2': '#2ca02c', 'TS3': '#d62728'}
    #ROC Curves
    ax1.grid(True)
    for name, preds in all_predictions.items():
        if name in all_labels:
            fpr, tpr, _ = roc_curve(all_labels[name], preds)
            roc_auc = results[name]['metrics']['roc_auc']
            display_name = name_mapping.get(name, name)
            ax1.plot(fpr, tpr, label=f'{display_name} (AUC = {roc_auc:.3f})', 
                    color=colors.get(display_name, '#1f77b4'), 
                    linewidth=2.0)
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    #PR Curves
    ax2.grid(True)
    for name, preds in all_predictions.items():
        if name in all_labels:
            precision, recall, _ = precision_recall_curve(all_labels[name], preds)
            pr_auc = results[name]['metrics']['pr_auc']
            display_name = name_mapping.get(name, name)
            ax2.plot(recall, precision, label=f'{display_name} (PR-AUC = {pr_auc:.3f})', 
                    color=colors.get(display_name, '#1f77b4'),
                    linewidth=2.0)
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(loc='lower left')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'combined_curves.png'), dpi=600, bbox_inches='tight')
    plt.close('all')

def test_model(model, test_loader, device):
    if test_loader is None or len(test_loader) == 0:
        raise ValueError("Test loader is empty or None")
    
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            num_batches += 1
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs.view(-1), batch.y.float())
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    if num_batches == 0:
        raise ValueError("No batches were processed during testing")
        
    avg_loss = total_loss / num_batches
    return np.array(all_preds), np.array(all_labels), avg_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plots_dir = os.path.join(os.getcwd(), 'saved_plots')
    os.makedirs(plots_dir, exist_ok=True)
    model_path = 'best_model.pth'
    try:
        dataset_mapping = get_dataset_mapping(model_path)
        with open(os.path.join(plots_dir, 'dataset_mapping.json'), 'w') as f:
            json.dump({'dataset_mapping': dataset_mapping}, f, indent=4)

        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)  
        data_result = dataload()
        logging.getLogger().setLevel(old_level)  
        
        (train_dataset, val_dataset, test_dataset, 
         ts1_dataset, ts2_dataset, ts3_dataset,
         num_node_features, num_edge_features,
         num_node_types, num_edge_types, class_weights,
         smiles_config) = data_result
        datasets = [test_dataset, ts1_dataset, ts2_dataset, ts3_dataset]
        try:
            old_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)
            loaders = create_optimized_dataloaders(datasets)
            logging.getLogger().setLevel(old_level)
            if not isinstance(loaders, dict) or len(loaders) == 0:
                raise ValueError("create_optimized_dataloaders didn't return valid loaders")
                
        except Exception as e:
            dataset_dict = {
                'test': test_dataset,
                'ts1': ts1_dataset,
                'ts2': ts2_dataset,
                'ts3': ts3_dataset
            }
            loaders = {name: torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
                      for name, dataset in dataset_dict.items() if dataset is not None and len(dataset) > 0}

        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get('model_config', {
            'in_dim': num_node_features,
            'hidden_dim': 256,
            'num_layers': 8,
            'num_heads': 8,
            'dropout': 0,
            'num_classes': 1,
            'num_node_types': num_node_types,
            'num_edge_types': num_edge_types,
            'processing_steps': 4,
            'use_Momentum': False  # Set to False to avoid contrastive learning components
        })
        model = MolecularNetwork(**model_config).to(device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        fixed_state_dict = fix_state_dict_keys(state_dict, use_contrastive=model_config.get('use_Momentum', False))
        missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
        
        model.eval()
        results = {}
        all_predictions = {}
        all_labels = {}
        loader_names = list(loaders.keys())
        display_names = ['Main Test', 'TS1', 'TS2', 'TS3']
        for i, (loader_name, loader) in enumerate(loaders.items()):
            if not loader:
                continue
                
            display_name = display_names[i] if i < len(display_names) else loader_name
            
            try:
                predictions, labels, loss = test_model(model, loader, device)
                metrics = compute_metrics(labels, predictions)
                results[display_name] = {'metrics': metrics, 'loss': loss}
                all_predictions[display_name] = predictions
                all_labels[display_name] = labels
                
                print(f"\n{display_name} Results:")
                print(f"Loss: {loss:.4f}")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"PR-AUC: {metrics['pr_auc']:.4f}")
                
            except Exception as e:
                logging.error(f"Error processing {loader_name} dataset: {str(e)}")
                continue
        save_results(results, all_predictions, all_labels, plots_dir)
        logging.info(f"\nTesting completed. Results saved in {plots_dir}")
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()