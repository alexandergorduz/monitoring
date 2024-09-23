import numpy as np
import pandas as pd
from typing import Dict



class PredictorsMonitor:
    """
    Class PredictorsMonitor for creating predictors monitor object.
    """


    def __init__(self, bins_amt: int = 10) -> None:
        """
        Initialize the PredictorsMonitor class.
        
        Args:
            bins_amt (int, optional): bins amount for numerical predictors.
        """

        self.bins_amt = bins_amt
        self.etalon_stat = None
    

    def fit(self, data: pd.DataFrame, checks: Dict = None) -> None:
        """
        Performs a fitting etalon data.
        
        Args:
            data (pd.DataFrame): etalon DataFrame with predictors.
            checks (Dict, optional): custom checks for predictors.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data should be a Pandas DataFrame instance."
            )
        
        self.preds = [pred for pred in data.columns if data[pred].dtype in ['int8', 'int16', 'int32', 'int64', 'float8', 'float16', 'float32', 'float64', 'object', 'category']]

        if len(self.preds) == 0:
            raise Exception(
                "There are no suitable predictors."
            )
        
        self.preds_types = {pred: ("NUMERICAL" if data[pred].dtype in ['int8', 'int16', 'int32', 'int64', 'float8', 'float16', 'float32', 'float64'] else "CATEGORY") for pred in self.preds}

        self.checks = Dict.fromkeys([f'{pred}__PSI' for pred in self.preds], 0.2)
        self.checks.update(Dict.fromkeys([f'{pred}__NA_PERC' for pred in self.preds], 0.1))

        if checks is not None:

            if not isinstance(checks, Dict):
                raise TypeError(
                    "Checks should be a Dict instance."
                )
            
            if not all(check in self.checks for check in checks):
                raise Exception(
                    "Some incomed checks unacceptable."
                )
            
            self.checks.update(checks)
        
        self.etalon_stat = {}

        for (pred, sr) in data.items():

            pred_stat = {}

            av_amt = sr.notna().sum()
            na_amt = sr.isna().sum()

            pred_stat['NA_PERC'] = round(na_amt / (av_amt + na_amt), 5)

            pred_stat['BINS'] = {}

            if sr.dtype in ['int8', 'int16', 'int32', 'int64', 'float8', 'float16', 'float32', 'float64']:

                bins = np.linspace(0.0, 1.0, self.bins_amt + 1)
                bins = sr.quantile(bins).values
                bins = np.unique(bins)
                bins = np.column_stack([bins[:-1], bins[1:]]).tolist()

                pred_stat['BINS_RANGES'] = bins

                for i, (left, right) in enumerate(bins):

                    if i == 0:
                        b_num = round(sr[sr <= right].shape[0] / av_amt, 5)
                    elif i == len(bins) - 1:
                        b_num = round(sr[left < sr].shape[0] / av_amt, 5)
                    else:
                        b_num = round(sr[(left < sr) & (sr <= right)].shape[0] / av_amt, 5)
                    
                    pred_stat['BINS'][f'B_{i + 1}'] = b_num.item()
            
            else:

                bins = sr.unique()

                for binn in bins:

                    if pd.isna(binn):
                        b_num = round(na_amt / (av_amt + na_amt), 5)
                    else:
                        b_num = round(sr[sr == binn].shape[0] / (av_amt + na_amt), 5)
                    
                    pred_stat['BINS'][f'{binn}'] = b_num.item()
            
            self.etalon_stat[pred] = pred_stat
    

    def monitor(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs a monitoring test data and creating summary DataFrame with monitoring inforamtion.
        
        Args:
            data (pd.DataFrame): test DataFrame with predictors to monitor.
        
        Returns:
            pd.DataFrame: summary DataFrame with monitoring information.
        """

        return self._check_stats(self.etalon_stat, self.get_test_stat(data[self.preds]))
    

    def get_test_stat(self, data: pd.DataFrame) -> Dict:
        """
        Performs a creating summary dictionary of test data in perspective of etalon data.
        
        Args:
            data (pd.DataFrame): test DataFrame with predictors to monitor.
        
        Returns:
            Dict: summary dictionary of test data in perspective of etalon data.
        """

        if self.etalon_stat is None:
            raise Exception(
                "PredictorsMonitor is not fitted yet."
            )
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data should be a Pandas DataFrame instance."
            )
        
        if not all(pred in data.columns for pred in self.preds):
            raise Exception(
                "Different predictors between etalon and test DataFrame's."
            )
        
        test_stat = {}

        for (pred, sr) in data.items():

            pred_stat = {}

            av_amt = sr.notna().sum()
            na_amt = sr.isna().sum()

            pred_stat['NA_PERC'] = round(na_amt / (av_amt + na_amt), 5)

            pred_stat['BINS'] = {}

            if sr.dtype in ['int8', 'int16', 'int32', 'int64', 'float8', 'float16', 'float32', 'float64']:

                bins = self.etalon_stat[pred]['BINS_RANGES']

                for i, (left, right) in enumerate(bins):

                    if i == 0:
                        b_num = round(sr[sr <= right].shape[0] / av_amt, 5)
                    elif i == len(bins) - 1:
                        b_num = round(sr[left < sr].shape[0] / av_amt, 5)
                    else:
                        b_num = round(sr[(left < sr) & (sr <= right)].shape[0] / av_amt, 5)
                    
                    pred_stat['BINS'][f'B_{i + 1}'] = b_num.item()
            
            else:

                bins = sr.unique()

                for binn in bins:

                    if pd.isna(binn):
                        b_num = round(na_amt / (av_amt + na_amt), 5)
                    else:
                        b_num = round(sr[sr == binn].shape[0] / (av_amt + na_amt), 5)
                    
                    pred_stat['BINS'][f'{binn}'] = b_num.item()
            
            test_stat[pred] = pred_stat
        
        return test_stat
    

    def _check_stats(self, etalon: Dict, test: Dict) -> pd.DataFrame:
        """
        Performs a creating summary DataFrame with monitoring information.
        
        Args:
            etalon (Dict): summary dictionary of etalon data.
            test (Dict): summary dictionary of test data in perspective of etalon data.
        
        Returns:
            pd.DataFrame: summary DataFrame with monitoring information.
        """

        res = []

        for pred in self.preds:

            pred_res = []

            row_tmplt = {'PRED_NAME': pred, 'PRED_TYPE': self.preds_types[pred], 'CHECK_STATE': 'OK'}

            etalon_bins = set(etalon[pred]['BINS'].keys())
            test_bins = set(test[pred]['BINS'].keys())
            inter_bins = list(etalon_bins.intersection(test_bins))

            if len(inter_bins) == 0:

                psi = None
            
            else:

                a = np.array([etalon[pred]['BINS'][binn] + 1e-10 for binn in inter_bins])
                b = np.array([test[pred]['BINS'][binn] + 1e-10 for binn in inter_bins])
                psi = round(np.sum((a - b) * np.log(a / b)), 5)
            
            row = row_tmplt.copy()
            row['CHECK_TYPE'] = 'PSI'
            row['CHECK_VALUE'] = f'{psi}'
            if psi is None or np.isnan(psi) or psi > self.checks[f'{pred}__PSI']:
                row['CHECK_STATE'] = 'NOT_OK'
            pred_res.append(row)

            etalon_na_perc = etalon[pred]['NA_PERC']
            test_na_perc = test[pred]['NA_PERC']

            row = row_tmplt.copy()
            row['CHECK_TYPE'] = 'NA_PERC'
            row['CHECK_VALUE'] = f'{test_na_perc}'
            if abs(etalon_na_perc - test_na_perc) > self.checks[f'{pred}__NA_PERC']:
                row['CHECK_STATE'] = 'NOT_OK'
            pred_res.append(row)

            if self.preds_types[pred] == "CATEGORY":

                row = row_tmplt.copy()
                row['CHECK_TYPE'] = 'NEW_VAL'
                if test_bins - etalon_bins:
                    row['CHECK_VALUE'] = f'{list(test_bins - etalon_bins)}'.replace('[', '| ').replace(']', ' |').replace(', ', ' | ')
                    row['CHECK_STATE'] = 'NOT_OK'
                else:
                    row['CHECK_VALUE'] = f'{None}'
                pred_res.append(row)

                row = row_tmplt.copy()
                row['CHECK_TYPE'] = 'NO_VAL'
                if etalon_bins - test_bins:
                    row['CHECK_VALUE'] = f'{list(etalon_bins - test_bins)}'.replace('[', '| ').replace(']', ' |').replace(', ', ' | ')
                    row['CHECK_STATE'] = 'NOT_OK'
                else:
                    row['CHECK_VALUE'] = f'{None}'
                pred_res.append(row)
            
            res.extend(pred_res)
        
        return pd.DataFrame(res, columns=['PRED_NAME', 'PRED_TYPE', 'CHECK_TYPE', 'CHECK_VALUE', 'CHECK_STATE'])



class PredictionsMonitor:
    """
    Class PredictionsMonitor for creating predictions monitor object.
    """


    def __init__(self, task: str = "classification", bins_amt: int = 10) -> None:
        """
        Initialize the PredictionsMonitor class.
        
        Args:
            task (str, optional): task that PredictionsMonitor works on, allowed: classification, regression.
            bins_amt (int, optional): bins amount for predictions.
        """

        if task not in ["classification", "regression"]:
            raise Exception(
                f"Only 'classification' and 'regression' allowed as tasks, got '{task}'."
            )
        
        self.task = task
        self.bins_amt = bins_amt
        self.etalon_stat = None

        if task == "classification":

            self.prefix = "CLASS"
        
        elif task == "regression":

            self.prefix = "REGRESSOR"
    

    def fit(self, data: np.ndarray, checks: Dict = None) -> None:
        """
        Performs a fitting etalon data.
        
        Args:
            data (np.ndarray): etalon ndarray with predictions.
            checks (Dict, optional): custom checks for predictions.
        """

        if not isinstance(data, np.ndarray):
            raise TypeError(
                "Input data should be a Numpy ndarray instance."
            )
        
        if self.task == "classification" and data.ndim != 2:
            raise Exception(
                f"For classification task PredictionsMonitor expects ndarray with 2 dimentions, got {data.ndim}."
            )
        elif self.task == "regression" and data.ndim not in [1, 2]:
            raise Exception(
                f"For regression task PredictionsMonitor expects ndarray with 1 or 2 dimentions, got {data.ndim}."
            )
        
        if np.isnan(data).any() or np.isinf(data).any():
            raise Exception(
                "Input data contains missing or infinity values."
            )
        
        if data.ndim == 1:

            data = data.reshape(-1, 1)
        
        self.ch_amt = data.shape[1]

        self.checks = {}

        if self.task == "classification":

            self.checks['ALL__MEAN_ENTHROPY'] = 0.1
        
        elif self.task == "regression":

            self.checks.update(Dict.fromkeys([f'{self.prefix}_{ch_num}__MEAN' for ch_num in range(self.ch_amt)], 0.1))
        
        self.checks.update(Dict.fromkeys([f'{self.prefix}_{ch_num}__PSI' for ch_num in range(self.ch_amt)], 0.2))
        self.checks.update(Dict.fromkeys([f'{self.prefix}_{ch_num}__OUTL_PERC' for ch_num in range(self.ch_amt)], 0.1))

        if checks is not None:

            if not isinstance(checks, Dict):
                raise TypeError(
                    "Checks should be a Dict instance."
                )
            
            if not all(check in self.checks for check in checks):
                raise Exception(
                    "Some incomed checks unacceptable."
                )
            
            self.checks.update(checks)
        
        av_amt = data.shape[0]

        self.etalon_stat = {}

        if self.task == "classification":

            self.etalon_stat['MEAN_ENTHROPY'] = round(np.mean(-np.sum(data * np.log(data + 1e-10), axis=1)), 5)
        
        for ch_num in range(self.ch_amt):

            ch = data[:, ch_num]

            ch_stat = {}

            mean = ch.mean()
            std = ch.std()

            if self.task == "regression":

                ch_stat['MEAN'] = round(mean, 5)
            
            left = mean - 3 * std
            right = mean + 3 * std

            ch_stat['OUTL_PERC'] = round(ch[(ch < left) | (ch > right)].shape[0] / av_amt, 5)

            bins = np.linspace(0.0, 1.0, self.bins_amt + 1)
            bins = np.quantile(ch, bins)
            bins = np.unique(bins)
            bins = np.column_stack([bins[:-1], bins[1:]]).tolist()

            ch_stat['BINS_RANGES'] = bins

            ch_stat['BINS'] = {}

            for i, (left, right) in enumerate(bins):

                if i == 0:
                    b_num = round(ch[ch <= right].shape[0] / av_amt, 5)
                elif i == len(bins) - 1:
                    b_num = round(ch[left < ch].shape[0] / av_amt, 5)
                else:
                    b_num = round(ch[(left < ch) & (ch <= right)].shape[0] / av_amt, 5)
                
                ch_stat['BINS'][f'B_{i + 1}'] = b_num
            
            self.etalon_stat[f'{self.prefix}_{ch_num}'] = ch_stat
    

    def monitor(self, data: np.ndarray) -> pd.DataFrame:
        """
        Performs a monitoring test data and creating summary DataFrame with monitoring information.
        
        Args:
            data (np.ndarray): test ndarray with predictions to monitor.
        
        Returns:
            pd.DataFrame: summary DataFrame with monitoring information.
        """

        return self._check_stats(self.etalon_stat, self.get_test_stat(data))
    

    def get_test_stat(self, data: np.ndarray) -> Dict:
        """
        Performs a creating summary dictionary of test data in perspective of etalon data.
        
        Args:
            data (np.ndarray): test ndarray with predictions to monitor.
        
        Returns:
            Dict: summary dictionary of test data in perspective of etalon data.
        """

        if self.etalon_stat is None:
            raise Exception(
                "PredictionsMonitor is not fitted yet."
            )
        
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "Input data should be a Numpy ndarray instance."
            )
        
        if self.task == "classification" and data.ndim != 2:
            raise Exception(
                f"For classification task PredictionsMonitor expects ndarray with 2 dimentions, got {data.ndim}."
            )
        elif self.task == "regression" and data.ndim not in [1, 2]:
            raise Exception(
                f"For regression task PredictionsMonitor expects ndarray with 1 or 2 dimentions, got {data.ndim}."
            )
        
        if np.isnan(data).any() or np.isinf(data).any():
            raise Exception(
                "Input data contains missing of infinity values."
            )
        
        if data.ndim == 1:

            data = data.reshape(-1, 1)
        
        if self.task == "classification" and self.ch_amt != data.shape[1]:
            raise Exception(
                f"PredictionsMonitor expects ndarray with {self.ch_amt} classes, got {data.shape[1]}."
            )
        elif self.task == "regression" and self.ch_amt != data.shape[1]:
            raise Exception(
                f"PredictionsMonitor expects ndarray with {self.ch_amt} regressors, got {data.shape[1]}."
            )
        
        av_amt = data.shape[0]

        test_stat = {}

        if self.task == "classification":

            test_stat['MEAN_ENTHROPY'] = round(np.mean(-np.sum(data * np.log(data + 1e-10), axis=1)), 5)
        
        for ch_num in range(self.ch_amt):

            ch = data[:, ch_num]

            ch_stat = {}

            mean = ch.mean()
            std = ch.std()

            if self.task == "regression":

                ch_stat['MEAN'] = round(mean, 5)
            
            left = mean - 3 * std
            right = mean + 3 * std

            ch_stat['OUTL_PERC'] = round(ch[(ch < left) | (ch > right)].shape[0] / av_amt, 5)

            bins = self.etalon_stat[f'{self.prefix}_{ch_num}']['BINS_RANGES']

            ch_stat['BINS'] = {}

            for i, (left, right) in enumerate(bins):

                if i == 0:
                    b_num = round(ch[ch <= right].shape[0] / av_amt, 5)
                elif i == len(bins) - 1:
                    b_num = round(ch[left < ch].shape[0] / av_amt, 5)
                else:
                    b_num = round(ch[(left < ch) & (ch <= right)].shape[0] / av_amt, 5)
                
                ch_stat['BINS'][f'B_{i + 1}'] = b_num
            
            test_stat[f'{self.prefix}_{ch_num}'] = ch_stat
        
        return test_stat
    

    def _check_stats(self, etalon: Dict, test: Dict) -> pd.DataFrame:
        """
        Performs a creating summary DataFrame with monitoring information.
        
        Args:
            etalon (Dict): summary dictionary of etalon data.
            test (Dict): summary dictionary of test data in perspective of etalon data.
        
        Returns:
            pd.DataFrame: summary DataFrame with monitoring information.
        """

        res = []

        if self.task == "classification":

            etalon_mean_enthropy = etalon['MEAN_ENTHROPY']
            test_mean_enthropy = test['MEAN_ENTHROPY']

            row = {f'{self.prefix}_NAME': 'ALL', 'CHECK_TYPE': 'MEAN_ENTHROPY', 'CHECK_VALUE': test_mean_enthropy, 'CHECK_STATE': 'OK'}
            if abs(etalon_mean_enthropy - test_mean_enthropy) > etalon_mean_enthropy * self.checks['ALL__MEAN_ENTHROPY']:
                row['CHECK_STATE'] = 'NOT_OK'
            
            res.append(row)
        
        for ch_num in range(self.ch_amt):

            ch_res = []

            row_tmplt = {f'{self.prefix}_NAME': f'{self.prefix}_{ch_num}', 'CHECK_STATE': 'OK'}

            etalon_bins = set(etalon[f'{self.prefix}_{ch_num}']['BINS'].keys())
            test_bins = set(test[f'{self.prefix}_{ch_num}']['BINS'].keys())
            inter_bins = list(etalon_bins.intersection(test_bins))

            if len(inter_bins) == 0:

                psi = None
            
            else:

                a = np.array([etalon[f'{self.prefix}_{ch_num}']['BINS'][binn] + 1e-10 for binn in inter_bins])
                b = np.array([test[f'{self.prefix}_{ch_num}']['BINS'][binn] + 1e-10 for binn in inter_bins])
                psi = round(np.sum((a - b) * np.log(a / b)), 5)
            
            row = row_tmplt.copy()
            row['CHECK_TYPE'] = 'PSI'
            row['CHECK_VALUE'] = psi
            if psi is None or np.isnan(psi) or psi > self.checks[f'{self.prefix}_{ch_num}__PSI']:
                row['CHECK_STATE'] = 'NOT_OK'
            ch_res.append(row)

            etalon_outl_perc = etalon[f'{self.prefix}_{ch_num}']['OUTL_PERC']
            test_outl_perc = test[f'{self.prefix}_{ch_num}']['OUTL_PERC']

            row = row_tmplt.copy()
            row['CHECK_TYPE'] = 'OUTL_PERC'
            row['CHECK_VALUE'] = test_outl_perc
            if abs(etalon_outl_perc - test_outl_perc) > self.checks[f'{self.prefix}_{ch_num}__OUTL_PERC']:
                row['CHECK_STATE'] = 'NOT_OK'
            ch_res.append(row)

            if self.task == "regression":

                etalon_mean = etalon[f'{self.prefix}_{ch_num}']['MEAN']
                test_mean = test[f'{self.prefix}_{ch_num}']['MEAN']

                row = row_tmplt.copy()
                row['CHECK_TYPE'] = 'MEAN'
                row['CHECK_VALUE'] = test_mean
                if abs(etalon_mean - test_mean) > abs(etalon_mean) * self.checks[f'{self.prefix}_{ch_num}__MEAN']:
                    row['CHECK_STATE'] = 'NOT_OK'
                ch_res.append(row)
            
            res.extend(ch_res)
        
        return pd.DataFrame(res, columns=[f'{self.prefix}_NAME', 'CHECK_TYPE', 'CHECK_VALUE', 'CHECK_STATE'])