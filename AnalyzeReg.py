import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from matplotlib.patches import Patch

def load_experiment_results(experiment_path):
    """
    Load results from JSON and CSV files in an experiment folder
    
    Args:
        experiment_path (str): Path to experiment folder
    
    Returns:
        dict: Loaded results
    """
    results = {}
    # print(f"\t\tFilePath: Loading results from {experiment_path}")
    
    # Try to load JSON results
    json_path = os.path.join(experiment_path, 'results.json')
    # print(f"\t\tFilePath: Looking for JSON at {json_path}")
    if os.path.exists(json_path):
        # print(f"\t\tFilePath: Found JSON file: {json_path}")
        with open(json_path, 'r') as f:
            results = json.load(f)
    # else:
        # print(f"\t\tFilePath: JSON file not found at {json_path}")
    
    # Check for CSV files
    csv_path = os.path.join(experiment_path, 'epoch_data.csv')
    # print(f"\t\tFilePath: Looking for CSV at {csv_path}")
    if os.path.exists(csv_path):
        # print(f"\t\tFilePath: Found CSV file: {csv_path}")
        results['epoch_data'] = pd.read_csv(csv_path)
    # else:
        # print(f"\t\tFilePath: CSV file not found at {csv_path}")
    
    # print(f"\t\tFilePath: Results keys found: {list(results.keys()) if results else 'None'}")
    return results

def create_epoch_plots(results, experiment_path):
    """
    Create plots for training, validation, and test metrics
    
    Args:
        results (dict): Experiment results
        experiment_path (str): Path to experiment folder
    """    
    # Check if epoch data exists
    if 'epoch_data' not in results:
        print(f"No epoch data found for {experiment_path}")
        return
    
    df = results['epoch_data']
    
    plt.figure(figsize=(15, 6))
    
    # Loss Subplot
    plt.subplot(1, 3, 1)
    plt.title('Loss per Epoch')
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Subplot
    plt.subplot(1, 3, 2)
    plt.title('Accuracy per Epoch')
    plt.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy')
    plt.plot(df['Epoch'], df['Validation Accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Metrics Comparison Subplot
    plt.subplot(1, 3, 3)
    metrics = {
        'Test Loss': results.get('test_loss', 0),
        'Test Accuracy': results.get('test_accuracy', 0)
    }
    
    # Create bar plot
    bars = plt.bar(list(metrics.keys()), list(metrics.values()))
    plt.title('Final Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Value')
    
    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, 
                    f'{height:.4f}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Add computation time as text at the bottom
    computation_time = results.get('computation_time', 0)
    plt.figtext(0.5, 0.01, 
        f'Computation Time: {int(computation_time // 3600)} Hrs {int((computation_time % 3600) // 60)} Min {int(computation_time % 60)} Sec', 
        ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    # Save the plot
    plot_path = os.path.join(experiment_path, 'epoch_analysis.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    
    plt.close()

def get_reg_directories(base_dir, noise_level):
    """
    Find all regularization directories for a given noise level
    
    Args:
        base_dir (str): Base directory to search in
        noise_level (float): Noise level to match
    
    Returns:
        dict: Dictionary mapping regularization types to their directories
    """
    reg_dirs = {}
    base_dir_name = f"Results_{noise_level}"
    
    # print(f"\t\tFilePath: Looking for base directory: {base_dir_name}")
    # Look for base directory
    if os.path.isdir(base_dir_name):
        reg_dirs['None'] = base_dir_name
        # print(f"\t\tFilePath: Found base directory: {base_dir_name}")
    # else:
        # print(f"\t\tFilePath: Base directory not found: {base_dir_name}")
    
    # Look for regularization directories
    for reg_type in ['L1', 'L2', 'DR', 'ES']:
        reg_dir_name = f"Results_{noise_level}_{reg_type}"
        # print(f"\t\tFilePath: Looking for reg directory: {reg_dir_name}")
        if os.path.isdir(reg_dir_name):
            reg_dirs[reg_type] = reg_dir_name
            # print(f"\t\tFilePath: Found reg directory: {reg_dir_name}")
        # else:
            # print(f"\t\tFilePath: Reg directory not found: {reg_dir_name}")
    
    # print(f"\t\tFilePath: Found directories: {reg_dirs}")
    return reg_dirs

def find_base_experiments(base_dir, dataset):
    """
    Find base experiments (without regularization) for a dataset
    
    Args:
        base_dir (str): Directory to search in
        dataset (str): Dataset name to filter by
    
    Returns:
        dict: Dictionary mapping experiment names to experiment paths
    """
    base_experiments = {}
    # print(f"\t\tFilePath: Searching for base experiments in: {base_dir} for dataset: {dataset}")
    
    # Look for experiments in the base directory
    if not os.path.isdir(base_dir):
        # print(f"\t\tFilePath: Base directory not found: {base_dir}")
        return base_experiments
        
    for experiment in os.listdir(base_dir):
        experiment_path = os.path.join(base_dir, experiment)
        # print(f"\t\tFilePath: Evaluating path: {experiment_path}")
        
        # Skip if not a directory or is a comparison directory
        if not os.path.isdir(experiment_path) or experiment.startswith('!Compare'):
            # print(f"\t\tFilePath: Skipping (not a directory or comparison dir): {experiment_path}")
            continue
        
        # Check if it's a base experiment and contains dataset name
        if not experiment.startswith('base_') or dataset.lower() not in experiment.lower():
            # print(f"\t\tFilePath: Skipping (not a base exp or doesn't contain dataset): {experiment}")
            continue
        
        # Add to dictionary
        base_experiments[experiment] = experiment_path
        # print(f"\t\tFilePath: Added base experiment: {experiment} at {experiment_path}")
    
    # print(f"\t\tFilePath: Found base experiments: {base_experiments}")
    return base_experiments

def compare_regularization_experiments(noise_level, dataset):
    """
    Compare experiments with different regularization techniques against base experiments
    
    Args:
        noise_level (float): Noise level to analyze
        dataset (str): Dataset name to filter by
    """
    # Find all relevant directories
    # print(f"\t\tFilePath: Finding directories for noise level: {noise_level}")
    reg_dirs = get_reg_directories(".", noise_level)
    
    if 'None' not in reg_dirs:
        # print(f"\t\tFilePath: Base directory Results_{noise_level} not found. Skipping analysis.")
        return
        
    # Create directory for comparison results
    compare_reg_path = os.path.join('Reg_Compare', f'{dataset.lower()}_{noise_level}')
    os.makedirs(compare_reg_path, exist_ok=True)
    # print(f"\t\tFilePath: Created comparison directory: {compare_reg_path}")
    
    # Find base experiments
    # print(f"\t\tFilePath: Looking for base experiments in {reg_dirs['None']}")
    base_experiments = find_base_experiments(reg_dirs['None'], dataset)
    
    if not base_experiments:
        # print(f"\t\tFilePath: No base experiments found for {dataset} in {reg_dirs['None']}. Skipping analysis.")
        return
    
    # Group regularization experiments by base name
    for base_exp_name, base_exp_path in base_experiments.items():
        # print(f"\t\tFilePath: Processing base experiment: {base_exp_name} at {base_exp_path}")
        # Initialize data structure to store experiment results
        reg_comparisons = {}
        
        # Load base experiment results
        base_results = load_experiment_results(base_exp_path)
        # print(f"\t\tFilePath: Base results loaded: {bool(base_results)}")
        if not base_results:
            # print(f"\t\tFilePath: No results found for base experiment {base_exp_name}. Skipping.")
            continue
            
        # Add base experiment to comparisons
        reg_comparisons['None'] = {
            'experiment_name': base_exp_name,
            'reg_type': 'None',  # String for display
            'test_loss': base_results.get('test_loss', 0),
            'test_accuracy': base_results.get('test_accuracy', 0),
            'computation_time': base_results.get('computation_time', 0),
            'epoch_data': base_results.get('epoch_data')
        }
        
        # Find and load regularization experiment results
        for reg_type, reg_dir in reg_dirs.items():
            if reg_type == 'None':
                continue  # Skip base directory, already processed
                
            # print(f"\t\tFilePath: Looking for matching experiments in {reg_dir}")
            # Look for matching experiment in the regularization directory
            for experiment in os.listdir(reg_dir):
                experiment_path = os.path.join(reg_dir, experiment)
                # print(f"\t\tFilePath: Evaluating reg experiment: {experiment_path}")
                
                # Skip if not a directory or is a comparison directory
                if not os.path.isdir(experiment_path) or experiment.startswith('!Compare'):
                    # print(f"\t\tFilePath: Skipping (not a directory or comparison dir): {experiment_path}")
                    continue
                
                # Modified check: Look for experiments that contain the dataset name and are base experiments
                if not (dataset.lower() in experiment.lower() and experiment.startswith('base_')):
                    # print(f"\t\tFilePath: Skipping (doesn't contain dataset or not base exp): {experiment}")
                    continue
                
                # Load results
                results = load_experiment_results(experiment_path)
                # print(f"\t\tFilePath: Results loaded for {experiment_path}: {bool(results)}")
                if not results:
                    continue
                
                # Add to comparisons
                reg_comparisons[reg_type] = {
                    'experiment_name': experiment,
                    'reg_type': reg_type,
                    'test_loss': results.get('test_loss', 0),
                    'test_accuracy': results.get('test_accuracy', 0),
                    'computation_time': results.get('computation_time', 0),
                    'epoch_data': results.get('epoch_data')
                }
                # print(f"\t\tFilePath: Added reg experiment to comparisons: {reg_type}")
        
        # Create comparison DataFrame
        # print(f"\t\tFilePath: Number of reg comparisons found: {len(reg_comparisons)}")
        if len(reg_comparisons) <= 1:
            # print(f"\t\tFilePath: No regularization experiments found for {base_exp_name}. Skipping comparison.")
            continue
            
        # Rest of function remains the same...
        comparison_data = []
        for reg_type, exp_data in reg_comparisons.items():
            comparison_data.append({
                'Regularization': reg_type,
                'Test Loss': exp_data['test_loss'],
                'Test Accuracy': exp_data['test_accuracy'],
                'Computation Time': exp_data['computation_time']
            })
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison data to CSV
        base_name_clean = base_exp_name.replace('/', '_').replace('\\', '_')
        csv_path = os.path.join(compare_reg_path, f'{base_name_clean}_reg_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        # print(f"\t\tFilePath: Saved comparison CSV to {csv_path}")
        
        # Create bar plots for each metric
        metrics = ['Test Loss', 'Test Accuracy', 'Computation Time']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Determine if lower is better for this metric
            lower_is_better = metric in ['Test Loss', 'Computation Time']
            
            # Sort by metric value
            if lower_is_better:
                comparison_df_sorted = comparison_df.sort_values(by=metric)
            else:
                comparison_df_sorted = comparison_df.sort_values(by=metric, ascending=False)
            
            # Highlight the base (no regularization) and best performer
            colors = []
            for reg_type in comparison_df_sorted['Regularization']:
                if reg_type == 'None':
                    colors.append('#1f77b4')  # Blue for base
                elif reg_type == comparison_df_sorted['Regularization'].iloc[0]:
                    colors.append('#2ca02c')  # Green for best
                else:
                    colors.append('#ff7f0e')  # Orange for others
            
            # Create bar plot
            bars = plt.bar(comparison_df_sorted['Regularization'], comparison_df_sorted[metric], color=colors)
            
            plt.title(f'{metric} Comparison for {dataset} - {base_exp_name} (Noise: {noise_level})')
            plt.xlabel('Regularization Type')
            plt.ylabel(metric)
            
            # Add value annotations
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, 
                        f'{height:.4f}', 
                        ha='center', va='bottom')
            
            # Add legend
            legend_elements = [
                Patch(facecolor='#1f77b4', label='No Regularization'),
                Patch(facecolor='#2ca02c', label='Best Performer'),
                Patch(facecolor='#ff7f0e', label='Other Regularization')
            ]
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(compare_reg_path, f'{base_name_clean}_{metric.lower().replace(" ", "_")}.png')
            plt.savefig(plot_path)
            # print(f"\t\tFilePath: Saved plot to {plot_path}")
            plt.close()

def create_overall_comparison(datasets, noise_levels):
    """
    Create overall comparison of regularization effectiveness across datasets and noise levels
    
    Args:
        datasets (list): List of datasets to analyze
        noise_levels (list): List of noise levels to analyze
    """
    # Create overall comparison directory
    overall_dir = os.path.join('Reg_Compare', '!Compare_Reg_Overall')
    os.makedirs(overall_dir, exist_ok=True)
    
    # Prepare data structure to store best regularization for each dataset and noise level
    best_reg_data = []
    
    for noise in noise_levels:
        # Find all regularization directories for this noise level
        reg_dirs = get_reg_directories(".", noise)
        
        if 'None' not in reg_dirs:
            continue
            
        for dataset in datasets:
            # Find base experiments
            base_experiments = find_base_experiments(reg_dirs['None'], dataset)
            
            for base_exp_name, base_exp_path in base_experiments.items():
                # Load base experiment results
                base_results = load_experiment_results(base_exp_path)
                if not base_results:
                    continue
                
                # Store base metrics
                base_metrics = {
                    'test_loss': base_results.get('test_loss', 0),
                    'test_accuracy': base_results.get('test_accuracy', 0)
                }
                
                # Find and load regularization experiment results
                reg_results = []
                for reg_type, reg_dir in reg_dirs.items():
                    if reg_type == 'None':
                        continue  # Skip base directory
                        
                    # Look for matching experiment in the regularization directory
                    for experiment in os.listdir(reg_dir):
                        experiment_path = os.path.join(reg_dir, experiment)
                        
                        # Skip if not a directory
                        if not os.path.isdir(experiment_path) or experiment.startswith('!Compare'):
                            continue
                        
                        # Skip if doesn't match dataset filter or isn't a base experiment
                        if not experiment.lower().endswith('_' + dataset.lower()) or not experiment.startswith('base_'):
                            continue
                        
                        # Load results
                        results = load_experiment_results(experiment_path)
                        if not results:
                            continue
                        
                        # Add to results list
                        reg_results.append({
                            'reg_type': reg_type,
                            'test_loss': results.get('test_loss', 0),
                            'test_accuracy': results.get('test_accuracy', 0)
                        })
                
                # Find best regularization (lowest test loss)
                if reg_results:
                    best_reg = min(reg_results, key=lambda x: x['test_loss'])
                    
                    # Calculate improvements
                    loss_improvement = base_metrics['test_loss'] - best_reg['test_loss']
                    accuracy_improvement = best_reg['test_accuracy'] - base_metrics['test_accuracy']
                    
                    # Add to overall data
                    best_reg_data.append({
                        'Dataset': dataset,
                        'Noise': noise,
                        'Base Experiment': base_exp_name,
                        'Best Reg': best_reg['reg_type'],
                        'Base Test Loss': base_metrics['test_loss'],
                        'Best Reg Test Loss': best_reg['test_loss'],
                        'Loss Improvement': loss_improvement,
                        'Base Test Accuracy': base_metrics['test_accuracy'],
                        'Best Reg Test Accuracy': best_reg['test_accuracy'],
                        'Accuracy Improvement': accuracy_improvement
                    })
    
    # Create overall comparison DataFrame
    if best_reg_data:
        overall_df = pd.DataFrame(best_reg_data)
        
        # Save to CSV
        overall_df.to_csv(os.path.join(overall_dir, 'regularization_comparison.csv'), index=False)
        
        # Create summary plots
        # Plot 1: Count of best regularization type by dataset
        plt.figure(figsize=(12, 6))
        reg_counts = overall_df['Best Reg'].value_counts()
        plt.bar(reg_counts.index, reg_counts.values)
        plt.title('Best Regularization Type Frequency')
        plt.xlabel('Regularization Type')
        plt.ylabel('Count')
        plt.savefig(os.path.join(overall_dir, 'best_reg_frequency.png'))
        plt.close()
        
        # Plot 2: Average improvement by regularization type
        plt.figure(figsize=(12, 6))
        avg_improvements = overall_df.groupby('Best Reg')['Loss Improvement'].mean()
        plt.bar(avg_improvements.index, avg_improvements.values)
        plt.title('Average Loss Improvement by Regularization Type')
        plt.xlabel('Regularization Type')
        plt.ylabel('Average Loss Improvement')
        plt.savefig(os.path.join(overall_dir, 'avg_loss_improvement.png'))
        plt.close()
        
        # Plot 3: Average accuracy improvement by regularization type
        plt.figure(figsize=(12, 6))
        avg_acc_improvements = overall_df.groupby('Best Reg')['Accuracy Improvement'].mean()
        plt.bar(avg_acc_improvements.index, avg_acc_improvements.values)
        plt.title('Average Accuracy Improvement by Regularization Type')
        plt.xlabel('Regularization Type')
        plt.ylabel('Average Accuracy Improvement')
        plt.savefig(os.path.join(overall_dir, 'avg_accuracy_improvement.png'))
        plt.close()
        
        # Plot 4: Heatmap of best regularization by dataset and noise
        plt.figure(figsize=(10, 8))
        
        # Create pivot table counting occurrence of each reg type for each dataset/noise combo
        heatmap_data = pd.crosstab(
            index=[overall_df['Dataset'], overall_df['Noise']], 
            columns=overall_df['Best Reg']
        ).fillna(0)
        
        # Find most frequent reg type for each dataset/noise
        best_reg_heatmap = pd.DataFrame(index=heatmap_data.index)
        best_reg_heatmap['Best Reg'] = heatmap_data.idxmax(axis=1)
        
        # Convert to format suitable for heatmap
        datasets = sorted(best_reg_heatmap.index.get_level_values(0).unique())
        noise_levels = sorted(best_reg_heatmap.index.get_level_values(1).unique())
        
        heatmap_array = np.zeros((len(datasets), len(noise_levels)), dtype=object)
        for i, dataset in enumerate(datasets):
            for j, noise in enumerate(noise_levels):
                try:
                    heatmap_array[i, j] = best_reg_heatmap.loc[(dataset, noise), 'Best Reg']
                except KeyError:
                    heatmap_array[i, j] = 'N/A'
        
        # Manual way to create heatmap with text values
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(np.zeros_like(heatmap_array, dtype=float), cmap='viridis')
        
        # Add text annotations
        for i in range(len(datasets)):
            for j in range(len(noise_levels)):
                text = ax.text(j, i, heatmap_array[i, j],
                              ha="center", va="center", color="black")
        
        # Set tick labels
        ax.set_xticks(np.arange(len(noise_levels)))
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_xticklabels([f"Noise: {n}" for n in noise_levels])
        ax.set_yticklabels(datasets)
        
        plt.title('Best Regularization Type by Dataset and Noise Level')
        plt.tight_layout()
        plt.savefig(os.path.join(overall_dir, 'best_reg_heatmap.png'))
        plt.close()

def main():
    """
    Analyze regularization experiments and compare with base (no regularization)
    """
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Analyze regularization experiment results')
    parser.add_argument('--noise', type=str, default='all',
                      help='Noise level for results directory (choices: 0.15, 0.25, 0.5, all) (default: all)')
    parser.add_argument('--dataset', type=str, default='all',
                     help='Dataset to use (choices: MNIST, CIFAR10, CIFAR100, STL10, all) (default: all)')
    args = parser.parse_args()
    
    # Define available noise levels and datasets
    noise_levels = [0.15, 0.25, 0.5]
    datasets = ['MNIST', 'CIFAR10', 'CIFAR100', 'STL10']
    
    # Determine which noise levels to process
    if args.noise.lower() == 'all':
        noise_to_process = noise_levels
    else:
        try:
            noise_value = float(args.noise)
            if noise_value in noise_levels:
                noise_to_process = [noise_value]
            else:
                print(f"Invalid noise value: {args.noise}. Using default 'all'.")
                noise_to_process = noise_levels
        except ValueError:
            print(f"Invalid noise value: {args.noise}. Using default 'all'.")
            noise_to_process = noise_levels
    
    # Determine which datasets to process
    if args.dataset.lower() == 'all':
        datasets_to_process = datasets
    elif args.dataset in datasets:
        datasets_to_process = [args.dataset]
    else:
        print(f"Invalid dataset: {args.dataset}. Using default 'all'.")
        datasets_to_process = datasets
    
    # Process each noise level
    for noise in noise_to_process:
        print(f"{'='*75}")
        print(f"\tProcessing regularization experiments for noise level: {noise}")
        
        # Process each dataset
        for dataset in datasets_to_process:
            print(f"{'='*75}")
            print(f"\tPROCESSING NOISE LEVEL: {noise}\t\tDATASET: {dataset}")
            print(f"{'='*75}")
            
            # Compare regularization experiments
            compare_regularization_experiments(noise, dataset)
    
    # Create overall comparison across all datasets and noise levels
    # create_overall_comparison(datasets_to_process, noise_to_process)
    
    print(f"{'='*75}")
    print("\tAnalysis complete! Check Reg_Compare directorie for results.")
    print(f"{'='*75}")

if __name__ == "__main__":
    main()