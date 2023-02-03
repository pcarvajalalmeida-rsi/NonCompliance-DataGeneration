# Time series Data Generator

import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn     #gaussian
from numpy.random import randint   #integers
import yaml
import hvplot.pandas
from dateutil.relativedelta import relativedelta
from datetime import datetime
import random

from dstools.datautils.datagen import random_datetimes, Source
from dstools.config.baseconfig import YmlConfig
from dstools.sqlutils.sqlconfig import SQLConfig
from dstools.sqlutils.util import DbConnectGenerator


yml = YmlConfig('./config/', read_all=True, ext='yaml')
yml_headers = yml.lowerkeys().keys()

seed=1
rng = np.random.default_rng(seed=seed)

def generate_random_id(n=8, seed=1, rng=None):
    """
    This function returns a number of n digits, which is to be used as an ID
  
    Arg:
    n: an integer 
    """
    start = 10 ** (n - 1)
    end = (10 ** n) - 1
    
    if rng==None:
        rng = np.random.default_rng(seed=seed)
  
    return rng.integers(start, end)


class gen_rand_periods():
    def __init__(self, yml=None, last_date='2022-12-31', 
                 max_nperiods_back = 40, 
                 freq='Q', prob_inactive=0.1, seed=1, 
                 src_file=None, outcomes_col=None, 
                 export_csv=False, debug=False, **args):
        """
        The class generates n random periods for one time series. The lenght n is itself a random number of periods. The random lenght period has up to max_nperiods_back records. 
        A proportion of prob_inactive companies are inactive. Inactive means that its last record will be before the last period signaled by the last_date.
        When the generated company is active, its last period will correspond to the last_date.
    A yaml configuration file can be given, with an appropiate header that identifies the parameter association (ie. 'period').
        When a source file with outcomes is not provided, a uniform distribution is assigned for the number of periods to be given. The number will go between the max number of periods, and 1/4 of that.
       
        
        Args:
        yml: a dictionary containing the parameters for the get_yml() function. In our regular usage, will be the ['period'] component of a YmlConfig() object.
        last_date: a string with the date of the last desired period for the data. ie'2022-12-31'
        src_file: is the file that contains in a columns (the `outcomes_col`) a list of outcomes for the number of periods in a sample. For example, the column may have the following values: [20,20,31,42,50] which will mean that out of 5 outcomes, 20 periods appear twice, and 31, 42 and 50 appear just one. That means the probability that will be applied will be 2/5 for 20, and 1/5 for the rest.
        """
        self.last_date  = pd.Timestamp(last_date)
        self.max_nperiods_back = max_nperiods_back
        self.freq = freq
        self.prob_inactive = prob_inactive
        self.yml = yml
        self.seed = seed
        self.src_file = src_file
        self.export_csv = False
        self.outcomes_col = outcomes_col
        self.debug = debug
        
    def get_yml(self):
        # get the config data
         
        self.last_date = pd.Timestamp(self.yml['last_date'])
        self.max_nperiods_back = self.yml['max_nperiods_back']
        self.freq = self.yml['freq']
        self.prob_inactive = self.yml['prob_inactive']
        self.src_file = self.yml['src_file']
        self.outcomes_col = self.yml['outcomes_col']
        
    def generate_periods(self, end, n, freq):
        """
        This method generates periods that are not random. Given the end of the period, and the number of periods, the function returns a timeseries of n periods of frequency `freq`, which ends in end. 
        For example, for end = '12-31-2022', n=4, and freq='Q', it will generate 4 quarters of data: 2022Q1, 2022Q2, 2022Q3, 2022Q4.
        
        Args:
        freq: str. "Q", "M" or "Y" for quarter, monthly or annual
        end: a date (Timestamp) which corresponds to the last period end (it will be the more recent part of the period, not the oldest.)
        n: an integer. The number of periods to generate.
        """
        if self.debug: print("enter generate_periods with end", end)
        
        end = pd.Period(end, freq=freq)
        start = end - n + 1
        
        return pd.period_range(start, end, freq=freq)
    
    
    def date_n_periods_back(self, last_date, n_back, freq=None):        
        if freq==None: 
            freq=self.freq
        
        last_date = pd.Timestamp(last_date)
        if freq == 'Q':
            delta = relativedelta(months= -n_back*3)
        elif freq == 'M':
            delta = relativedelta(months=-n_back)
        elif freq == 'Y':
            delta = relativedelta(years=-n_back)
        else:
            return "Invalid frequency. Please use 'Q' for quarters or 'M' for months."

        self.start_date = start_date = pd.Timestamp((last_date + delta).strftime("%Y-%m-%d"))
        return start_date
  
        
    def get_sample(self):
        """
        It generates a list of consecutive periods of frequence `freq`.
        The periods are associated to active or inactive companies. There is a probability "self.prob_inactive" that the company is not active. This inactive companies will have a period that ends before the provided `last_date`.
        """        
        # First, we generates a 0 or 1 according to the probability of inactive
        rng = np.random.default_rng(seed = self.seed)
        inactive = rng.binomial(1, self.prob_inactive)

        if inactive==1:
            if self.debug: print("inactive flag")
            self.start_date = self.date_n_periods_back(last_date = self.last_date, n_back = self.max_nperiods_back, freq=self.freq)
            self.last_date = random_datetimes(start = self.start_date , end = self.last_date)
        
        # Generate a random number of periods for the generated time series
        if self.src_file==None:
            self.sample_n = rng.integers(self.max_nperiods_back//4, 
                                         self.max_nperiods_back)
        
        else:
            n_source = pd.read_csv(self.src_file)
            sample_n = Source(outcomes=n_source[self.outcomes_col])
            sample_n.calculate_weights()
            sample_n_dic = sample_n.get_sample()
            self.sample_n=sample_n_dic['n']
        
        self.periods = self.generate_periods(end = self.last_date, n=self.sample_n, freq=self.freq)
        
        return self.periods



def generate_random_numbers(distribution, params, n, seed):
    """
    This function generates random numbers of different distributions.
    
    Args:
    distribution: str. Possible: 'normal', 'beta', 'uniform', 'binomial'
    params: a dictionary with the relevant parameters for each distribition. For example:
              for a beta:    {'a': 8, 'b': 8},
              for a uniform: {'low': 0, 'high': 1},
              for a normal:  {'mean': 250000, 'std': 100000}
              for a binomial:{'n':5,'p': 0.6}
    
    """
    
    rng = np.random.default_rng(seed = seed)
    if distribution == 'normal':
        return rng.normal(params['mean'], params['std'], n)
    elif distribution == 'beta':
        return rng.beta(params['a'], params['b'], n)
    elif distribution == 'uniform':
        return rng.uniform(params['low'], params['high'], n)
    elif distribution == 'binomial':
        return rng.binomial(params['n'], params['p'], n)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")



def normalize_rows(matrix):
    '''
    This function will transform each row of a matrix of proportions so the sum of the proportions sum up to one.
    In the case a all rows sum up to zero, it will keep it as zero.
    
    matrix: an np array
    '''
    
    # Cases of vectors instead of matrices
    if len(matrix.shape)==1:
        if matrix.sum()==0:
            pass
        else: matrix = matrix / matrix.sum()
        
    # Case of a real matrix 
    else:    
        for i in range(matrix.shape[0]):
        # Get the sum of the current row
            row_sum = np.sum(matrix[i,:])

            # If the row sum is not zero, normalize the row
            if row_sum != 0:
                matrix[i,:] = matrix[i,:] / row_sum
    return matrix


class gen_random_proportions():
    """
    Class to generate a random sample of n records of multiple 
    ions or
    [net income, effective tax rate and aggregate deductions],
    or any set of proportions, normalized (sum up to 1) or not normalized.
    Result is given in a dataframe.

    Args:
    features_p: is a dictionary which the name of the feature as keys, and 
    the probability of that features to be different than 0 values
        for example, for a case of 3 deductions: {'deduc_1': 0.9, 'deduc_2': 0.001369, 'deduc_3': 0.017539}

    distribution: is a dictionary  with the distributions of each feature
        for example: {'deduc_1': 'uniform', 'deduc_2': 'uniform', 'deduc_3': 'uniform'}, 

    parameters: are the parameters of the distributions above.
        for example: {'deduc_1': {'low': 0, 'high': 1}, 'deduc_2': {'low': 0, 'high': 1}, 'deduc_3': {'low': 0, 'high': 1}}

    Distribution supported and its parameters are defined in the function generate_random_numbers
    """
    
    def __init__(self, yml, rng=None, features_p=None, distribution=None, parameters=None,
                  n = 40, seed = 5, normalized = True, export_csv = False, debug=False, include_agg=True,
                  #always_non_neg=None, 
                 global_always_non_neg=True):

        self.yml = yml
        self.rng = rng
        self.features_p = features_p
        self.distribution = distribution
        self.parameters = parameters
        
        self.n = n
        self.seed = seed        
        self.normalized = normalized
        self.export_csv  = export_csv
        self.debug=debug
        self.include_agg=include_agg
        #self.always_non_neg=always_non_neg
        self.global_always_non_neg=global_always_non_neg
        

        
    def get_yml(self):
        """
        This class read the configutation from the yml class associated to the object
        """
        self.features_p = self.yml['features_p']
        self.distribution = self.yml['distribution']
        self.parameters = self.yml['parameters']
        self.include_agg = self.yml['include_agg']
        self.normalized = self.yml['normalized']
        #self.always_non_neg = yml['always_non_neg']
        #@TODO: case when only some of the variables are non-negative self.global_always_non_neg = yml['global_always_non_neg']
        

    def get_sample(self):
        """
        Steps:
        For each deduction:
        1. Generate random 0s and 1s based on the binomial probability of having the feature
        2. Generate random numbers based on the distribution of no-zero values provided in the distribution
        3. Multiply the vector from step 1 with the vector from step 2
        4. Normalize the vector so to sum up to one if there are positive values for the features, or sum up to zero if there are not
        """
        is_first_iter = 1

        for feature, p in self.features_p.items():
            rng = np.random.default_rng(seed = self.seed)
            # step 1
            got_feature = rng.binomial(1, p, self.n)
            # step 2
            random_column = generate_random_numbers(self.distribution[feature], self.parameters[feature], self.n, self.seed)
            # step 3
            final_column = got_feature * random_column
            
            if is_first_iter == False:   
                matrix = np.column_stack((matrix, final_column))

            if is_first_iter:
                matrix = final_column
                is_first_iter = False
            self.seed +=1
        if self.debug:
            print("matrix without normalization: \n",matrix)

        if self.global_always_non_neg:
            matrix[matrix<0]=0
            
        # for row, index, (key, non_neg) in zip(matrix, enumerate(self.always_non_neg.items())):
        #      if row[index]<0 and non_neg==True:
        #             row[index] = 0
           
        
        if self.normalized:
            matrix = normalize_rows(matrix)
            if self.debug:
                print("matrix normalized: \n",matrix)
            
        matrix_df = pd.DataFrame(matrix, columns = list(self.distribution.keys()))
        
        if self.include_agg:
            matrix_df['include_agg'] = matrix_df.sum(axis=1)
            
        #if self.export_csv:
            #file_name = list(matrix_df.columns)[0][0:5]
            #matrix_df.to_csv(f"{file_name}_seed_{self.seed}.csv")
        
        return matrix_df



class gen_money_ts():
    """
    This class generates time series of money, so it can be used to model income or other finanial time series
    """

    def __init__(self, yml, distribution=None, parameters=None, 
                  n = 40, seed = 5, export_csv = False, debug=False, 
                always_non_neg=None):

        self.distribution = distribution
        self.parameters = parameters
        self.n = n
        self.seed = seed
        self.export_csv  = export_csv
        self.debug=debug
        self.always_non_neg=always_non_neg
        self.yml = yml
 
        
    def get_yml(self):
        # get the config data
        self.distribution = self.yml['distribution']
        self.parameters = self.yml['parameters']
        self.always_non_neg = self.yml['always_non_neg']
            
    def get_sample(self):
        random_ts = generate_random_numbers(self.distribution, self.parameters, self.n, self.seed)
        if self.always_non_neg:
            random_ts = abs(random_ts)
        random_ts_df = pd.DataFrame(random_ts, columns=['income'])
        return round(random_ts_df,2)

     


class ObjectGenerator(object):
    
    """
    This object allows to create any object and populate it with data.
    It applies to objects that have:
        1) the method get_yml() defined, to populate their parameters
        2) the method get_sample() defined, to generate a sample of it
    The purpose is to allow easily the construction of data by using the combination of generators of time series columns or DataFrames, in an very suscint way, as in the function `gen_ts_record`.
    
    Args:
    1) Main ones when using yml configuration file
    `yml`: the YAML parameters specifics to the object
    `class_name`: the name of the class to be instantiated to generate the object (ie. gen_random_proportions)
    
    2) Args for when instantiated the object without yml files:
    `class_name` (same as before)
    `n`: number of rows in teh time series. Default to 40, which are ten years of quarters periods
    `seed`: integer that represents the seed for the random state
    `debug`: True if you want to see intermediate printings in the procecss. Default to False
    `include_agg`: Binary. In case of proportions, if you want to include the sum of the proportions as another column.
    `export_csv`: Binary. If you want to get a csv file with the time series generated. Default to False.
    
    
    """
    def __init__(self, yml, class_name = None,  #_type=TaxPayerType.individual, 
                 n=40, seed=None, debug=False, include_agg=True, export_csv=False):
        self.yml = yml
        self.class_name = class_name
        self.debug = debug
        self.seed = (np.random.randint(0,np.iinfo(np.int32).max) if seed is None else seed)
        self.export_csv = export_csv
        self.n = n
        self.include_agg = include_agg
   
    def populate_object(self):
        # Get the class object from the class_name parameter
        class_ = globals().get(self.class_name)

        # Instantiate the class
        data_obj = class_(yml=self.yml, n=self.n,seed=self.seed, 
                           export_csv=self.export_csv, debug=self.debug)
        data_obj.get_yml()
        ts_record_df = data_obj.get_sample()

        return ts_record_df




# def gen_ts_record_old(yml, seed=1, debug=False):
#     """
#     This function generates a record comprised by a concatenation of objects and possible transformation among them.
#     The steps are:
#     1. Generation of columns from objects
#     1.1. Index using the `gen_rand_periods()` class. This step defines the random length of the record.
#     1.2. Ids   using the `generate_random_id()` function.
#     1.3. Income using the `gen_money_ts()` class
#     1.4. Distribution of the income in tax paid, net income and aggregate deductions, using the `gen_random_proportions()` class. This distribution for each row of the record.
#     1.5. Distribution of deductions in every period (row) of the record
    
#     2. Tranformation: converting proportions to money and re-grouping/drops
#     2.1. Calculate tax_paid, aggregate deductions as amplyfying the proportion by the income
#     2.2. Calculate deductions as amplyfying the proportions by the income
    
    
#     3. Setting up the index as the element in 1.1 for all the dataframe record
    
    
#     Args:
#     yml: a yaml dictionary containing the key parameters for the objects: 'periods', 'income', and 'deducs'. For the required parameters, see the method get_yml() associated to each object
#     """
    

#     # GENERATION OF INDEX AND OBJECTS
#     # This can be made using a loop, but I prefer to see each of the components
   
#     # 1. Index periods
#     periods = ObjectGenerator(yml=yml.get('periods'), class_name='gen_rand_periods', seed=seed)
#     periods_index = periods.populate_object()
#     n = len(periods_index)
#     if debug: print(n, periods_index)

#     # 2. Business ID of id_digits
#     business_id = generate_random_id(n=8, seed=seed)
#     id_df = pd.DataFrame([business_id]*n, columns=['business_id'])
#     if debug: display(id_df)

#     # 3. Income - money
#     income = ObjectGenerator(yml=yml.get('income'), class_name='gen_money_ts', n=n, seed=seed)
#     income_df = income.populate_object()
#     if debug: display(income_df)

#     # 4. Distribution of the income: effective tax rate + aggregate deductions + net income
#     efftax_aggdeduc_netinc = ObjectGenerator(yml=yml.get('efftax_aggdeduc_netinc'), 
#                                              class_name='gen_random_proportions', 
#                                              n=n, seed=seed)
#     efftax_aggdeduc_netinc_df = efftax_aggdeduc_netinc.populate_object()
#     if debug: display(efftax_aggdeduc_netinc_df)

#     # 5. Distribution of deductions
#     deducs = ObjectGenerator(yml=yml.get('deducs'), 
#                              class_name='gen_random_proportions', 
#                              n=n, seed=seed
#     )
#     deducs_df = deducs.populate_object()
#     if debug: display(deducs_df)

#     # 2. TRANSFORMATIONS FROM THE RAW OBJECTS TO THE FINAL TIME SERIES OBJECT
#     # amplified proportions to money
#     # 2.1. tax_paid, aggregate deductions: from proportions to money
#     efftax_aggdeduc_netinc_df = income_df.values * efftax_aggdeduc_netinc_df
#     efftax_aggdeduc_df = efftax_aggdeduc_netinc_df.drop(columns=['net_income'])
#     mapper = {'effect_tax_rate': 'tax_paid'}
#     taxpaid_aggdeduc_df = efftax_aggdeduc_df.rename(columns=mapper)

#     # 2.2 Deductions from proportion to money
#     deducs_df = taxpaid_aggdeduc_df[['agg_deduc']].values * deducs_df
#     record_df = pd.concat([id_df,income_df, 
#                            taxpaid_aggdeduc_df, 
#                            deducs_df], 
#                           axis=1
#     )
#     record_df = round(record_df,2)

#     #3. Setting up the index
#     # Adding period index
#     record_df.index = periods_index
#     record_df.sort_index(ascending=False, inplace=True)

#     return record_df






def gen_ts_record_raw(yml, seed=1, debug=False):
    """
    This function generates a rawa record comprised by a concatenation of objects and possible transformation among them.
    The steps are:
    1. Generation of columns from objects
    1.1. Index using the `gen_rand_periods()` class. This step defines the random length of the record.
    1.2. Ids   using the `generate_random_id()` function.
    1.3. Income using the `gen_money_ts()` class
    1.4. Distribution of the income in tax paid, net income and aggregate deductions, using the `gen_random_proportions()` class. This distribution for each row of the record.
    1.5. Distribution of deductions in every period (row) of the record
    
    
    Args:
    yml: a yaml dictionary containing the key parameters for the objects: 'periods', 'income', and 'deducs'. For the required parameters, see the method get_yml() associated to each object
    
    Returns:
    A dictionary of dataframes and a period index.
    """
    
    # GENERATION OF INDEX AND OBJECTS
    # This can be made using a loop, but I prefer to see each of the components
    
    # Resetting random state so always gives same record
    # initial_state = random.getstate()
    # random.seed(seed)

    # Index periods
    periods = ObjectGenerator(yml=yml.get('periods'), class_name='gen_rand_periods', seed=seed)
    periods_index = periods.populate_object()
    n = len(periods_index)
    if debug: print(n, periods_index)

    # Business ID of id_digits
    business_id = generate_random_id(n=8, seed=seed)
    id_df = pd.DataFrame([business_id]*n, columns=['business_id'])
    if debug: display(id_df)

    # Income -money
    income = ObjectGenerator(yml=yml.get('income'), class_name='gen_money_ts', n=n, seed=seed)
    income_df = income.populate_object()
    if debug: display(income_df)

    # Decomposition of the income: effective tax rate + aggregate deductions + net income
    efftax_aggdeduc_netinc = ObjectGenerator(yml=yml.get('efftax_aggdeduc_netinc'), 
                                             class_name='gen_random_proportions', 
                                             n=n, seed=seed)
    efftax_aggdeduc_netinc_df = efftax_aggdeduc_netinc.populate_object()
    if debug: display(efftax_aggdeduc_netinc_df)

    # Decomposition of deductions
    deducs = ObjectGenerator(yml=yml.get('deducs'), class_name='gen_random_proportions', 
                             n=n, seed=seed)
    deducs_df = deducs.populate_object()
    if debug: display(deducs_df)

    raw_ts_objects = {'periods_index': periods_index, 
                      'id_df': id_df,
                      'income_df': income_df, 
                      'efftax_aggdeduc_netinc_df':efftax_aggdeduc_netinc_df, 
                      'deducs_df': deducs_df
                     }
        
    return raw_ts_objects


def transform_raw_ts_record(raw_ts_objects):
    """
    This function work together with  gen_ts_record_raw to transforma set of raw records.
    After the Generation of columns from objects, the transformation do as follows:
    
    2. Tranformation: converting proportions to money and re-grouping/drops
    2.1. Calculate tax_paid, aggregate deductions as amplyfying the proportion by the income
    2.2. Calculate deductions as amplyfying the proportions by the income
    
    
    3. Setting up the index as the element in 1.1 for all the dataframe record
    
    
    Args:
    raw_ts_objects which is a dictionary of raw dataframe objects. populated, that needs to be transformed
    For example:     {'periods_index': periods_index, 
                      'id_df': id_df,
                      'income_df': income_df, 
                      'efftax_aggdeduc_netinc_df':efftax_aggdeduc_netinc_df, 
                      'deducs_df': deducs_df
                     }

    """
    
    # Getting the values
    periods_index = raw_ts_objects['periods_index']
    id_df         = raw_ts_objects['id_df']
    efftax_aggdeduc_netinc_df = raw_ts_objects['efftax_aggdeduc_netinc_df']
    income_df     = raw_ts_objects['income_df']
    deducs_df     = raw_ts_objects['deducs_df']
    
    # 2. TRANSFORMATIONS FROM THE RAW OBJECTS TO THE FINAL TIME SERIES OBJECT
    # amplified proportions to money
    # 2.1. tax_paid, aggregate deductions: from proportions to money
   
    efftax_aggdeduc_netinc_df = income_df.values * efftax_aggdeduc_netinc_df
    efftax_aggdeduc_df = efftax_aggdeduc_netinc_df.drop(columns=['net_income'])
    mapper = {'effect_tax_rate': 'tax_paid'}
    taxpaid_aggdeduc_df = efftax_aggdeduc_df.rename(columns=mapper)

    # 2.2 Deductions from proportion to money
    deducs_df = taxpaid_aggdeduc_df[['agg_deduc']].values * deducs_df
    record_df = pd.concat([id_df,income_df, 
                           taxpaid_aggdeduc_df, 
                           deducs_df], 
                          axis=1
    )
    record_df = round(record_df,2)

    #3. adding index
    # Adding period index
    record_df.index = periods_index
    record_df.sort_index(ascending=False, inplace=True)

    #random.setstate(initial_state)
    return record_df


def gen_ts_record(yml, seed=1, debug=False):
    raw_ts_objects = gen_ts_record_raw(yml, seed=seed, debug=False)
    record_df = transform_raw_ts_record(raw_ts_objects)
    return (record_df)



class gen_ts_dataset():
    """
    This class generates a long dataset of time series, by concatenating one random time series after the other.
    
    Args:
    n_samples: int. Number of time series in the dataset. For example, in the case of non-compliant data, a time series corresponds to the historical records associated to one business and one location. One company with 40 records correspond to n_sample=1. Two companies, one with 10 records and the other with 50 corresponds to n_sample=2.

    seed: int. A initial seed for the random state of the generation
    """
    def __init__(self, yml, n_samples=50, seed=1, debug=False, export_csv=True):
        self.n_samples = n_samples
        self.seed = seed
        self.debug=debug
        self.export_csv=export_csv
        self.yml = yml
       
    def get_dataset(self):
        initial=1
        for k in range(0,self.n_samples):
            if self.debug: print('record', k, 'seed:', self.seed)
            new_record = gen_ts_record(yml=self.yml, seed=self.seed, debug=self.debug)
            if initial:
                self.dataset_df = new_record
                initial=0
            else:
                self.dataset_df = pd.concat([self.dataset_df, new_record], axis=0)
            self.seed+=1

        if self.export_csv: self.dataset_df.to_csv('new_dataset.csv')

        return self.dataset_df



dataset_df = gen_ts_dataset(yml=yml, n_samples=100, seed=1, debug=False, export_csv=True)
dataset_df = dataset_df.get_dataset()
dataset_df


