import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd

class SliceResult:

    def __init__(self, exp_name, keywords, kw2cat, category_column=None, is_final=False):
        '''
            df: pandas dataframe
            keywords: slice keywords
        '''
        self.exp_name = exp_name
        self.keywords = keywords
        self.kw2cat = kw2cat
        self.is_final = is_final
        self.category_column = category_column
        if is_final:
            self.df = pd.read_csv(f"result/{exp_name}/final_result.csv")
        else:
            self.df = pd.read_csv(f"result/{exp_name}/slice_result.csv")
        self.generate_ground_truth()
        self.generate_majority_vote()
        
    def generate_ground_truth(self):
        for kw in self.keywords:
            if self.category_column is not None:
                self.df[f'{kw}_gt'] = self.df[self.category_column] == self.kw2cat[kw]
            else:
                self.df[f'{kw}_gt'] = self.df[self.kw2cat[kw]]

    def generate_majority_vote(self):
        def majority_vote(row, cols):
            return int(row[cols].mean() > 0.5)
        for kw in self.keywords:
            if self.is_final:
                pred_cols = [col for col in self.df.columns if col.startswith(f'label_{kw}') and not col.endswith('meta')]
            else:
                pred_cols = [col for col in self.df.columns if col.startswith(f'{kw}_prompt') and not col.endswith('meta')]
            self.df[f'{kw}_maj'] = self.df.apply(lambda row: majority_vote(row, pred_cols), axis=1)
    
    def compute_stats(self, col, kw):
        df = self.df
        gt_col = df[f'{kw}_gt']

        def acc(col):
            return (df[col] == gt_col).mean()

        def pseudo_acc(col):
            return (df[col] == df[f'{kw}_maj']).mean()

        def precision(col):
            '''
            precision = TP / (TP + FP)
            '''
            return ((df[col] == 1) & (gt_col == 1)).sum() / (df[col] == 1).sum()

        def recall(col):
            '''
            recall = TP / (TP + FN)
            '''
            return ((df[col] == 1) & (gt_col == 1)).sum() / (gt_col == 1).sum()

        def f1(col):
            prec = precision(col)
            rec = recall(col)
            return 2 * prec * rec / (prec + rec)

        def positive(col):
            return (df[col] == 1).sum() / len(df)

        def negative(col):
            return (df[col] == 0).sum() / len(df)

        def positive_gt(col):
            return (gt_col == 1).sum() / len(df)

        return {
            'Acc': acc(col),
            'Precision': precision(col),
            'Recall': recall(col),
            'F1': f1(col),
            'Positive': positive(col),
            'Negative': negative(col),
            'Pseudo Acc': pseudo_acc(col),
            'Slice Fraction': positive_gt(col),
        }

    def compute_stats_all(self):
        result_table = pd.DataFrame(columns=['Acc', 'Precision', 'Recall', 'F1', 'Positive', 'Negative', 'Pseudo Acc', 'Slice Fraction', 'Prompt'])
        for idx, kw in enumerate(self.keywords):
            if self.is_final:
                pred_cols = [col for col in self.df.columns if col.startswith(f'label_{kw}') and not col.endswith('meta')]
            else:
                pred_cols = [col for col in self.df.columns if col.startswith(f'{kw}_prompt') and not col.endswith('meta')]
            prompts = pd.read_csv(f"result/{self.exp_name}/prompt_result_{idx}.csv")
            for n_col, col in enumerate(pred_cols):
                stats = self.compute_stats(col, kw)
                stats['Prompt'] = prompts.iloc[n_col][f'{kw}_prompt']
                result_table.loc[col] = stats
        self.result_table = result_table
        return result_table



if __name__ == "__main__":
    exp_name = 'bbq_llama_0'
    keywords = ['age', 'gender identity', 'disability status', 'nationality', 'religion']
    kw2cat = {
        'age': 'Age',
        'gender identity': 'Gender_identity',
        'disability status': 'Disability_status',
        'nationality': 'Nationality',
        'religion': 'Religion'
    }
    sliceResult = SliceResult(exp_name, keywords, kw2cat)
    table = sliceResult.compute_stats_all()
    print(table)

# categories = ['abortion', 'atheism', 'climate', 'feminist', 'hillary']
# keywords = {cat: cat for cat in categories}
# exp_name = 'tweets_t5_0'

# categories = ['high_school_biology', 'high_school_chemistry', 'high_school_psychology', 'high_school_macroeconomics', 'high_school_statistics']
# keywords = {cat: cat.replace('high_school_', '') for cat in categories}
# exp_name = 'mmlu_t5_0'

# categories = ['negation']
# keywords = {cat: cat for cat in categories}
# exp_name = 'superglue_t5_0'