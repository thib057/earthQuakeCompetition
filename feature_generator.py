import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


class FeatureGenerator(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        self.total_data = 0
        if self.dtype == 'train':
            self.filename = '../data/train.csv'
        else:
            submission = pd.read_csv('../data/sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, '../data/test/' + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64}, nrows = 500*150000)
            self.total_data = int(4*150000 / self.chunk_size)

            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values[-1]
                seg_id = 'train_' + str(counter)
                yield seg_id, x, y
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
                x = df.acoustic_data.values
                yield seg_id, x, -999

    def features(self,x, y, seg_id):
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id

        # create features here
        # for example:
        feature_dict['mean'] = np.mean(x)
        feature_dict['max'] = np.max(x)
        feature_dict['min'] = np.min(x)
        feature_dict['last_sample'] = x[-1]
        feature_dict['diff_val'] = x[-1] - x[0]
        return feature_dict

    def generate(self):
        feature_list = []
        res = Parallel(n_jobs=self.n_jobs,
                       backend='threading')(delayed(self.features)(x, y, s)
                                            for s, x, y in tqdm(self.read_chunks(), total=self.total_data))
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)