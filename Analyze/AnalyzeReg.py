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
    
    # Try to load JSON results
    json_path = os.path.join(experiment_path, 'results.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        print(f"\t\tFilePath: JSON file not found at {json_path}")
    
    # Check for CSV files
    csv_path = os.path.join(experiment_path, 'epoch_data.csv')
    if os.path.exists(csv_path):
        results['epoch_data'] = pd.read_csv(csv_path)
    else:
        print(f"\t\tFilePath: CSV file not found at {csv_path}")
    
    return results

def load_best_reg_parameter(experiment_path, reg_type):
    """
    Load best regularization parameter value from summary JSON file
    
    Args:
        experiment_path (str): Path to experiment folder
        reg_type (str): Regularization type (L1, L2, DR, ES)
    
    Returns:
        str: Parameter value as string, or None if not found
    """
    reg_type_lower = reg_type.lower()
    json_path = os.path.join(experiment_path, f'best_{reg_type_lower}_summary.json')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            summary_data = json.load(f)
            
        # Extract parameter based on regularization type
        if reg_type == 'L1':
            return str(summary_data.get('best_l1_value'))
        elif reg_type == 'L2':
            return str(summary_data.get('best_l2_value'))
        elif reg_type == 'DR':
            return str(summary_data.get('best_dropout_rate'))
        elif reg_type == 'ES':
            return str(summary_data.get('best_es_value'))
    
    return None

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

def get_reg_directories(noise_level):
    """
    Find all regularization directories for a given noise level
    
    Args:
        noise_level (float): Noise level to match
    
    Returns:
        dict: Dictionary mapping regularization types to their directories
    """
    reg_dirs = {}
    base_dir_name = f"../Results_{noise_level}"
    
    # Look for base directory
    if os.path.isdir(base_dir_name):
        reg_dirs['None'] = base_dir_name
    
    # Look for regularization directories
    for reg_type in ['L1', 'L2', 'DR', 'ES']:
        reg_dir_name = f"../Results_{noise_level}_{reg_type}"
        if os.path.isdir(reg_dir_name):
            reg_dirs[reg_type] = reg_dir_name
    
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
    
    # Look for experiments in the base directory
    if not os.path.isdir(base_dir):
        return base_experiments
        
    for experiment in os.listdir(base_dir):
        experiment_path = os.path.join(base_dir, experiment)
        
        # Skip if not a directory or is a comparison directory
        if not os.path.isdir(experiment_path) or experiment.startswith('!Compare'):
            continue

        # Check if it's a base experiment and exactly matches the dataset name
        if not experiment.startswith('base_'):
            continue
        exp_dataset = experiment[len('base_'):].split('_')[0].lower()
        if exp_dataset != dataset.lower():
            continue
        
        # Add to dictionary
        base_experiments[experiment] = experiment_path
    
    return base_experiments

def find_best_reg_experiments(reg_dir, reg_type, dataset, noise_level):
    """
    Find best regularization experiments for a dataset and noise level
    
    Args:
        reg_dir (str): Directory to search in
        reg_type (str): Regularization type
        dataset (str): Dataset name
        noise_level (float): Noise level
    
    Returns:
        dict: Dictionary mapping experiment names to experiment paths
    """
    best_experiments = {}
    
    # Check if directory exists
    if not os.path.isdir(reg_dir):
        return best_experiments
    
    # Create pattern to match
    pattern = f"!Best_{reg_type}_{dataset}_{noise_level}"
    
    for experiment in os.listdir(reg_dir):
        experiment_path = os.path.join(reg_dir, experiment)
        
        # Skip if not a directory
        if not os.path.isdir(experiment_path):
            continue
        
        # Check if it matches the pattern
        if experiment.startswith(pattern):
            best_experiments[experiment] = experiment_path
            break  # Only need one match
    
    return best_experiments

def compare_regularization_experiments(noise_level, dataset):
    """
    Compare experiments with different regularization techniques against base experiments
    
    Args:
        noise_level (float): Noise level to analyze
        dataset (str): Dataset name to filter by
    """
    # Find all relevant directories
    reg_dirs = get_reg_directories(noise_level)
    
    if 'None' not in reg_dirs:
        return
        
    # Create directory for comparison results
    compare_reg_path = os.path.join('../Reg_Compare', f'{dataset.lower()}_{noise_level}')
    os.makedirs(compare_reg_path, exist_ok=True)
    
    # Find base experiments
    base_experiments = find_base_experiments(reg_dirs['None'], dataset)
    
    if not base_experiments:
        return
    
    # Group regularization experiments by base name
    for base_exp_name, base_exp_path in base_experiments.items():
        # Initialize data structure to store experiment results
        reg_comparisons = {}
        
        # Load base experiment results
        base_results = load_experiment_results(base_exp_path)
        if not base_results:
            continue
            
        # Add base experiment to comparisons
        reg_comparisons['None'] = {
            'experiment_name': base_exp_name,
            'reg_type': 'None',  # String for display
            'test_loss': base_results.get('test_loss', 0),
            'test_accuracy': base_results.get('test_accuracy', 0),
            'computation_time': base_results.get('computation_time', 0),
            'epoch_data': base_results.get('epoch_data'),
            'display_name': 'None'  # New field for display name
        }
        
        # Find and load regularization experiment results
        for reg_type, reg_dir in reg_dirs.items():
            if reg_type == 'None':
                continue  # Skip base directory, already processed
            
            # Look for best regularization experiment
            best_experiments = find_best_reg_experiments(reg_dir, reg_type, dataset, noise_level)
            
            for experiment, experiment_path in best_experiments.items():
                # Load results
                results = load_experiment_results(experiment_path)
                if not results:
                    continue
                
                # Load best regularization parameter
                param_value = load_best_reg_parameter(experiment_path, reg_type)
                
                # Create display name with parameter value
                display_name = reg_type
                if param_value:
                    display_name = f"{reg_type} {param_value}"
                
                # Add to comparisons
                reg_comparisons[reg_type] = {
                    'experiment_name': experiment,
                    'reg_type': reg_type,
                    'test_loss': results.get('test_loss', 0),
                    'test_accuracy': results.get('test_accuracy', 0),
                    'computation_time': results.get('computation_time', 0),
                    'epoch_data': results.get('epoch_data'),
                    'display_name': display_name  # New field for display name
                }
        
        # Create comparison DataFrame
        if len(reg_comparisons) <= 1:
            continue
            
        comparison_data = []
        for reg_type, exp_data in reg_comparisons.items():
            comparison_data.append({
                'Regularization': reg_type,
                'Display Name': exp_data['display_name'],  # New field for display
                'Test Loss': exp_data['test_loss'],
                'Test Accuracy': exp_data['test_accuracy'],
                'Computation Time': exp_data['computation_time']
            })
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison data to CSV
        base_name_clean = base_exp_name.replace('/', '_').replace('\\', '_')
        csv_path = os.path.join(compare_reg_path, f'{base_name_clean}_reg_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        
        # When creating bar plots for each metric
        metrics = ['Test Loss', 'Test Accuracy', 'Computation Time']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Determine if lower is better for this metric
            lower_is_better = metric in ['Test Loss', 'Computation Time']
            
            # Create a copy of the DataFrame for custom sorting
            comparison_df_plot = comparison_df.copy()
            
            # Create a custom sort order: 'None' first, then others sorted by metric
            if lower_is_better:
                reg_sorted = comparison_df[comparison_df['Regularization'] != 'None'].sort_values(by=metric)['Regularization'].tolist()
            else:
                reg_sorted = comparison_df[comparison_df['Regularization'] != 'None'].sort_values(by=metric, ascending=False)['Regularization'].tolist()
                
            # Final sort order: 'None' first, then others
            sort_order = ['None'] + reg_sorted
            
            # Create a new category column with proper order
            comparison_df_plot['Order'] = comparison_df_plot['Regularization'].apply(lambda x: sort_order.index(x))
            comparison_df_plot = comparison_df_plot.sort_values('Order')
            
            # Get the best regularization method (not including 'None')
            reg_only_df = comparison_df[comparison_df['Regularization'] != 'None']
            if not reg_only_df.empty:
                if lower_is_better:
                    best_reg = reg_only_df.sort_values(by=metric).iloc[0]['Regularization']
                else:
                    best_reg = reg_only_df.sort_values(by=metric, ascending=False).iloc[0]['Regularization']
            else:
                best_reg = None
            
            # Highlight the base (no regularization) and best performer
            colors = []
            for reg_type in comparison_df_plot['Regularization']:
                if reg_type == 'None':
                    colors.append('#1f77b4')  # Blue for base
                elif reg_type == best_reg:
                    colors.append('#2ca02c')  # Green for best regularization
                else:
                    colors.append('#ff7f0e')  # Orange for others
            
            # Create bar plot with the custom order and display names
            bars = plt.bar(comparison_df_plot['Display Name'], comparison_df_plot[metric], color=colors)
            
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
                Patch(facecolor='#2ca02c', label='Best Regularization'),
                Patch(facecolor='#ff7f0e', label='Other Regularization')
            ]
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(compare_reg_path, f'{base_name_clean}_{metric.lower().replace(" ", "_")}.png')
            plt.savefig(plot_path)
            plt.close()

def create_overall_comparison(datasets, noise_levels):
    """
    Create overall comparison of regularization effectiveness across datasets and noise levels
    
    Args:
        datasets (list): List of datasets to analyze
        noise_levels (list): List of noise levels to analyze
    """
    # Create overall comparison directory
    overall_dir = os.path.join('../Reg_Compare', '!Compare_Reg_Overall')
    os.makedirs(overall_dir, exist_ok=True)
    
    # Prepare data structure to store best regularization for each dataset and noise level
    best_reg_data = []
    
    for noise in noise_levels:
        # Find all regularization directories for this noise level
        reg_dirs = get_reg_directories(noise)
        
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
                        
                    # Look for best regularization experiment
                    best_experiments = find_best_reg_experiments(reg_dir, reg_type, dataset, noise)
                    
                    for experiment, experiment_path in best_experiments.items():
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
    create_overall_comparison(datasets_to_process, noise_to_process)
    
    print(f"{'='*75}")
    print("\tAnalysis complete! Check Reg_Compare directories for results.")
    print(f"{'='*75}")

if __name__ == "__main__":
    main()