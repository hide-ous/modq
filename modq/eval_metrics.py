import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, \
    multilabel_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def _generate_plots(df, results, output_dir, show_plots=False):
    """
    Generates and optionally saves evaluation plots.

    Args:
        df (pd.DataFrame): DataFrame with evaluation data.
        results (dict): Dictionary containing evaluation results.
        output_dir (str): Directory to save the plots.
        show_plots (bool): Whether to display the plots.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Confusion Matrix across Rule Categories ---
    if 'category_confusion_matrix' in results:
        cm_category = results['category_confusion_matrix']
        categories = results['all_categories']
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_category, annot=True, fmt='d', cmap='Blues',
                    xticklabels=categories,
                    yticklabels=categories,
                    cbar=False)
        plt.xlabel('Predicted Category', fontsize=12)
        plt.ylabel('True Category', fontsize=12)
        plt.title('Confusion Matrix (Rule Categories)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_categories.png'))
        if show_plots:
            plt.show()

    # --- Confusion Matrix for the Binary Task ---
    y_true_binary = (df['true_safe']).astype(int)
    y_pred_binary = (df['predicted_safe']).astype(int)
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unsafe', 'Safe'], yticklabels=['Unsafe', 'Safe'],
                cbar=False)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Binary Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_binary.png'))
    if show_plots:
        plt.show()

    # --- Bar plots of prec, rec, ACC, f1 per rule category ---
    category_df = pd.DataFrame.from_dict(results['per_category'], orient='index')
    category_df.plot(kind='bar', figsize=(12, 6), colormap='viridis')
    plt.title('Performance Metrics per Rule Category', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.xlabel('Rule Category', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_metrics_bar.png'))
    if show_plots:
        plt.show()

    # --- Horizontal bar plots of top 10 best and worst predicted rules (F1) ---
    rule_f1 = pd.Series({k: v['f1'] for k, v in results['per_rule'].items()})
    top_10_best_rules = rule_f1.nlargest(10)
    top_10_worst_rules = rule_f1.nsmallest(10)

    if not top_10_best_rules.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(top_10_best_rules.index.astype(str), top_10_best_rules.values, color='green')
        plt.xlabel('F1 Score', fontsize=12)
        plt.ylabel('Rule Text', fontsize=12)
        plt.title('Top 10 Best Predicted Rules (F1)', fontsize=14)
        plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_10_best_rules.png'))
        if show_plots:
            plt.show()

    if not top_10_worst_rules.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(top_10_worst_rules.index.astype(str), top_10_worst_rules.values, color='red')
        plt.xlabel('F1 Score', fontsize=12)
        plt.ylabel('Rule Text', fontsize=12)
        plt.title('Top 10 Worst Predicted Rules (F1)', fontsize=14)
        plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_10_worst_rules.png'))
        if show_plots:
            plt.show()

    # --- Horizontal bar plots of top 10 best and worst predicted communities (F1) ---
    community_f1 = pd.Series({k: v['f1'] for k, v in results['per_community'].items()})
    top_10_best_communities = community_f1.nlargest(10)
    top_10_worst_communities = community_f1.nsmallest(10)

    if not top_10_best_communities.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(top_10_best_communities.index, top_10_best_communities.values, color='green')
        plt.xlabel('F1 Score', fontsize=12)
        plt.ylabel('Community', fontsize=12)
        plt.title('Top 10 Best Predicted Communities (F1)', fontsize=14)
        plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_10_best_communities.png'))
        if show_plots:
            plt.show()

    if not top_10_worst_communities.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(top_10_worst_communities.index, top_10_worst_communities.values, color='red')
        plt.xlabel('F1 Score', fontsize=12)
        plt.ylabel('Community', fontsize=12)
        plt.title('Top 10 Worst Predicted Communities (F1)', fontsize=14)
        plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_10_worst_communities.png'))
        if show_plots:
            plt.show()

    # --- Line plot of performance vs rule number (using F1 score) ---
    if results['per_rule']:
        rule_performance_df = pd.DataFrame.from_dict(results['per_rule'], orient='index')
        plt.figure(figsize=(10, 6))
        plt.plot(rule_performance_df.index, rule_performance_df['f1'], marker='o', linestyle='-', color='#4c78a8')
        plt.xlabel('Rule Number', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Performance vs Rule Number (F1)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_vs_rule_number.png'))
        if show_plots:
            plt.show()

    # --- Line plot of performance vs number of rule options (using Accuracy) ---
    performance_vs_options = df.groupby('n_rule_options')[['predicted_rule_n', 'true_rule_n']].apply(
        lambda x: accuracy_score(x['true_rule_n'], x['predicted_rule_n'])
    ).reset_index(name='accuracy')

    plt.figure(figsize=(10, 6))
    plt.plot(performance_vs_options['n_rule_options'], performance_vs_options['accuracy'], marker='o', linestyle='-', color='#d17a3c')
    plt.xlabel('Number of Rule Options', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Performance vs Number of Rule Options (Accuracy)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_options.png'))
    if show_plots:
        plt.show()



def evaluate_rule_classification(df, output_dir=None, output_plots=False, show_plots=False):
    """
    Computes and optionally saves/plots evaluation metrics for rule classification.
    Handles cases where true and predicted rule categories are lists.
    A rule is considered correctly predicted if there is any overlap
    between the true and predicted category lists.

    Args:
        df (pd.DataFrame): DataFrame with columns including: "comment_id", 'community',
            'n_rule_options', 'true_safe', 'predicted_safe', 'true_rule_text',
            'predicted_rule_text', 'true_rule_category', 'predicted_rule_category',
            'true_rule_n', 'predicted_rule_n'.  Assumes 'true_rule_category' and
            'predicted_rule_category' can be either single values or lists
            of category values.
        output_dir (str, optional): Path to the directory to save the
            evaluation results. Defaults to None.
        output_plots (bool, optional): Whether to generate and save plots.
            Defaults to False.

    Returns:
        dict: A dictionary containing the computed evaluation metrics.
    """

    # df.true_rule_category = df.true_rule_category.apply(eval)
    # df.predicted_rule_category = df.predicted_rule_category.apply(eval)
    df.true_safe = df.true_rule_text == 'Safe'
    df.predicted_safe = df.predicted_rule_text == 'Safe'
    results = {}

    # 1. Micro-averaged metrics (regardless of specific rules)
    y_true_micro = df['true_rule_n'].values
    y_pred_micro = df['predicted_rule_n'].values
    results['micro_accuracy'] = accuracy_score(y_true_micro, y_pred_micro)
    #print(classification_report(y_true_micro, y_pred_micro, zero_division=0))
    results['micro_accuracy']
    mlb = MultiLabelBinarizer().fit(pd.concat([df.true_rule_category, df.predicted_rule_category], axis=0))
    category_df = pd.DataFrame(
        classification_report(mlb.transform(df.true_rule_category), mlb.transform(df.predicted_rule_category),
                              target_names=mlb.classes_, output_dict=True)).T
    #print(category_df)

    macrof1 = category_df.loc["macro avg", "f1-score"]

    category_df
    
    ## comment out for reddit
    try:
        binary_df = pd.DataFrame(classification_report(df.true_safe, df.predicted_safe, target_names=['Not Safe', 'Safe', ], output_dict=True)).T
    except:
        binary_df = pd.DataFrame()
    #binary_df
    ###
    #binary_df = pd.DataFrame()
    
    confusion_matrices_per_category = multilabel_confusion_matrix(mlb.transform(df.true_rule_category),
                                                                  mlb.transform(df.predicted_rule_category))
    for cat, conf in zip(mlb.classes_, confusion_matrices_per_category):
        print(cat, '\n', conf)

    caf_conf_binary_df = pd.DataFrame(confusion_matrices_per_category.reshape((-1, 4)), index=mlb.classes_,
                                      columns=['tn', 'fp', 'fn', 'tp'])
    caf_conf_binary_df

    def create_mlcm(true_labels, predicted_labels, num_classes):
        """
        Creates the Multi-Label Confusion Matrix (MLCM).

        Args:
            true_labels: List of lists, where each inner list contains the true labels for an instance.
            predicted_labels: List of lists, where each inner list contains the predicted labels for an instance.
            num_classes: The number of classes in the multi-label classification problem.

        Returns:
            MLCM: A (num_classes + 1) x (num_classes + 1) numpy array representing the MLCM.
                  The last row is for 'No True Label' (NTL), and the last column is for 'No Predicted Label' (NPL).
        """

        mlcm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
        label2int = {v: k for k, v in
                     enumerate(
                         sorted(set(j for i in true_labels for j in i) | set(j for i in predicted_labels for j in i)))}

        for i in range(len(true_labels)):
            true_set = set(true_labels[i])
            predicted_set = set(predicted_labels[i])

            # Category 1: Predicted is a subset of True
            if predicted_set.issubset(true_set):
                true_positive_set = true_set.intersection(predicted_set)
                for label in true_positive_set:
                    # Ensure label is an integer before using it as an index
                    mlcm[label2int[label], label2int[label]] += 1  # True Positives
                if not true_set and not predicted_set:
                    mlcm[num_classes, num_classes] += 1  # No True Label and No Predicted Label

            # Category 2: True is a proper subset of Predicted
            elif true_set.issubset(predicted_set) and true_set != predicted_set:
                true_positive_set = true_set.intersection(predicted_set)
                incorrect_predicted_set = predicted_set.difference(true_set)
                for label in true_positive_set:
                    # Ensure label is an integer
                    mlcm[label2int[label], label2int[label]] += 1  # True Positives
                for true_label in true_set:
                    for incorrect_label in incorrect_predicted_set:
                        # Ensure labels are integers
                        mlcm[label2int[true_label], label2int[incorrect_label]] += 1  # False Negatives
                if not true_set:
                    for incorrect_label in incorrect_predicted_set:
                        # Ensure label is an integer
                        mlcm[num_classes, label2int[incorrect_label]] += 1

            # Category 3:  Intersection is not equal to True or Predicted
            elif not true_set.issubset(predicted_set) and not predicted_set.issubset(true_set):
                true_positive_set = true_set.intersection(predicted_set)
                incorrect_true_set = true_set.difference(predicted_set)
                incorrect_predicted_set = predicted_set.difference(true_set)
                for label in true_positive_set:
                    mlcm[label2int[label], label2int[label]] += 1
                for true_label in incorrect_true_set:
                    for incorrect_label in incorrect_predicted_set:
                        mlcm[label2int[true_label], label2int[incorrect_label]] += 1
        return mlcm, label2int

    mlcm, label2int = create_mlcm(df.true_rule_category, df.predicted_rule_category, len(mlb.classes_))
    lbls = [i[0] for i in sorted(label2int.items(), key=lambda x: x[1])]
    mlcm_df = pd.DataFrame(mlcm, index=lbls + ['NTL'], columns=lbls + ['NPL'])
    mlcm_df
    # 3. Metrics per rule category
    unique_categories = sorted(set(cat for categories in df['true_rule_category'] for cat in
                                   (categories if isinstance(categories, list) else [categories])))
    category_metrics = {}
    for category in unique_categories:
        y_true_cat = df['true_rule_category'].apply(
            lambda x: 1 if isinstance(x, list) and category in x else (1 if x == category else 0)).astype(int)
        y_pred_cat = df['predicted_rule_category'].apply(
            lambda x: 1 if isinstance(x, list) and category in x else (1 if x == category else 0)).astype(int)
        category_metrics[category] = {
            'precision': precision_score(y_true_cat, y_pred_cat, average='macro', zero_division=0),
            'recall': recall_score(y_true_cat, y_pred_cat, average='macro', zero_division=0),
            'f1': f1_score(y_true_cat, y_pred_cat, average='macro', zero_division=0),
            'accuracy': accuracy_score(y_true_cat, y_pred_cat, )
        }
    results['per_category_macro'] = category_metrics
    print("REAL MACRO")
    print(category_metrics)

    # 4. Metrics per rule
    unique_rules = sorted(df['true_rule_text'].unique())
    rule_metrics = {}
    for rule_text in unique_rules:
        y_true_rule = (df['true_rule_text'] == rule_text).astype(int)
        y_pred_rule = (df['predicted_rule_text'] == rule_text).astype(int)
        rule_metrics[rule_text] = {
            'precision': precision_score(y_true_rule, y_pred_rule, zero_division=0),
            'recall': recall_score(y_true_rule, y_pred_rule, zero_division=0),
            'f1': f1_score(y_true_rule, y_pred_rule, zero_division=0),
            'accuracy': accuracy_score(y_true_rule, y_pred_rule),
            'support': y_true_rule.sum()
        }
    results['per_rule'] = rule_metrics

    # 5. Metrics per community
    unique_communities = sorted(df['community'].unique())
    community_metrics = {}
    for community in unique_communities:
        df_community = df[df['community'] == community]
        y_true_comm = df_community['true_rule_n'].values
        y_pred_comm = df_community['predicted_rule_n'].values
        community_metrics[community] = {
            'precision': precision_score(y_true_comm, y_pred_comm, average='weighted', zero_division=0),
            'recall': recall_score(y_true_comm, y_pred_comm, average='weighted', zero_division=0),
            'f1': f1_score(y_true_comm, y_pred_comm, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_true_comm, y_pred_comm),
            'support': y_true_comm.sum()
        }

    results['per_community'] = community_metrics

    if output_dir:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, 'evaluation_results.xlsx')
        # Prepare data for CSV export

        rule_df = pd.DataFrame.from_dict(results['per_rule'], orient='index')
        community_df = pd.DataFrame.from_dict(results['per_community'], orient='index')
        macro_df = pd.DataFrame.from_dict(results['per_category_macro'], orient='index')

        with pd.ExcelWriter(output_csv_path) as writer:
            category_df.to_excel(writer, sheet_name='Per Category Metrics')
            macro_df.to_excel(writer, sheet_name='Per Category Metrics (Macro)')
            binary_df.to_excel(writer, sheet_name='Binary Metrics')
            rule_df.to_excel(writer, sheet_name='Per Rule Metrics')
            community_df.to_excel(writer, sheet_name='Per Community Metrics')
            mlcm_df.to_excel(writer, sheet_name='MultiLabel Confusion Matrix')
            caf_conf_binary_df.to_excel(writer, sheet_name='Binary Confusion Matrix')

        #print(f"\nResults saved to '{output_csv_path}'")

    # if output_dir:
    #     # Create the output directory if it doesn't exist
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_csv_path = os.path.join(output_dir, 'evaluation_results.xlsx')
    #     # Prepare data for CSV export
    #     category_df = pd.DataFrame.from_dict(results['per_category'], orient='index')
    #     rule_df = pd.DataFrame.from_dict(results['per_rule'], orient='index')
    #     community_df = pd.DataFrame.from_dict(results['per_community'], orient='index')
    #     overall_df = pd.DataFrame({
    #         'metric': ['micro_precision', 'micro_recall', 'micro_f1', 'micro_accuracy',
    #                    'macro_precision_category', 'macro_recall_category', 'macro_f1_category', 'macro_accuracy_category',
    #                    'binary_precision', 'binary_recall', 'binary_f1', 'binary_accuracy'],
    #         'value': [results['micro_precision'], results['micro_recall'], results['micro_f1'], results['micro_accuracy'],
    #                   results['macro_precision_category'], results['macro_recall_category'], results['macro_f1_category'], results['macro_accuracy_category'],
    #                   results['binary_precision'], results['binary_recall'], results['binary_f1'], results['binary_accuracy']]
    #     })
    #
    #     with pd.ExcelWriter(output_csv_path) as writer:
    #         overall_df.to_excel(writer, sheet_name='Overall Metrics', index=False)
    #         category_df.to_excel(writer, sheet_name='Per Category Metrics')
    #         rule_df.to_excel(writer, sheet_name='Per Rule Metrics')
    #         community_df.to_excel(writer, sheet_name='Per Community Metrics')
    #
    #     print(f"\nResults saved to '{output_csv_path}'")

    # if output_plots:
    #     _generate_plots(df, results, output_dir, show_plots=show_plots)

    return results, macrof1
