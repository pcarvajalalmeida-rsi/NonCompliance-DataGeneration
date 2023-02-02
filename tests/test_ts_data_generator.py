from ts_dataset_generator import (
    generate_random_id, gen_rand_periods,
    gen_random_proportions, normalize_rows
)
        
import pytest
import pandas as pd
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
from pandas import PeriodIndex

# Sample input for the test
yml = {
   'periods':{
       'last_date': '2022-12-31',
       'max_nperiods_back': 40,
       'freq': 'Q',
       'prob_inactive': 0.05,
       'src_file': 'Resources/81n_per_loc_and_bus.csv',
       'outcomes_col': 'n_per_each_loc_bus_id'
   },
   'efftax_aggdeduc_netinc':{
      'naics': 81,
      'size': 's',
      'include_agg': False,
      'normalized': True,
      'features_p': {'effect_tax_rate': 1, 'agg_deduc': 1, 'net_income': 1},
      'distribution': {'effect_tax_rate': 'normal',
                      'agg_deduc': 'beta',
                      'net_income': 'normal'
                    },
      'parameters': {
        'effect_tax_rate': {'mean': 0.15, 'std': 0.1},
        'agg_deduc': {'a': 2, 'b': 6},
        'net_income': {'mean': 0.67, 'std': 0.05}
                   },
      'global_always_non_neg': False,
      'always_non_neg': {
          'net_income': False,
          'effect_tax_rate': True,
          'agg_deduc': True}
   },  
    
   'income': {
      'naics': 81,
      'size': 's',
      'always_non_neg': True,
      'period': 'Q',
      'last_date': datetime.date(2022, 12, 31),
      'distribution': 'normal',
      'parameters': {'mean': 250000, 'std': 100000}
   },
    'deducs':{
        'naics': 81,
        'size': 's',
        'include_agg': False,
        'normalized': True,
        'global_always_non_neg': True,
        'features_p': {
                        'deduc_1': 0.9,
                        'deduc_2': 0.75,
                        'deduc_16': 0.001369,
                        'deduc_17': 0.017539
                        },
        'distribution': {
                        'deduc_1': 'beta',
                        'deduc_2': 'uniform',
                        'deduc_16': 'uniform',
                        'deduc_17': 'uniform'
        },
        'parameters': {
                  'deduc_1': {'a': 8, 'b': 8},
                  'deduc_2': {'low': 0, 'high': 1},
                  'deduc_16': {'low': 0, 'high': 1},
                  'deduc_17': {'low': 0, 'high': 1},
                  }
        }
}



seed = 1
rng  = np.random.default_rng(seed=seed)

def test_generate_random_id():
    # CASE PASSING RNG
    _id = generate_random_id(n=8, rng=rng)
    
    # Check the output is an integer
    assert isinstance(_id, np.int64)
    
    # Check output has eight digits
    assert len(str(_id)) == 8
    
    # CASE PASSING SEED
    _id = generate_random_id(n=10, seed=1)
    
    # Check the output is an integer
    assert isinstance(_id, np.int64)
    
    # Check it has ten digits
    assert len(str(_id)) == 10

 ########################################################
 #  TESTING PERIOD GENERATION - gen_rand_periods OBJECT #
 ########################################################
def test_gen_rand_periods_initialization():
    # INIT METHOD - DEFAULT VALUES
    obj = gen_rand_periods(yml['periods'])

    assert obj.last_date == pd.Timestamp('2022-12-31')
    assert obj.max_nperiods_back == 40
    assert obj.freq == 'Q'
    assert obj.prob_inactive == 0.1
    assert obj.yml == yml['periods']
    assert obj.seed == 1
    assert obj.src_file == None
    assert obj.export_csv == False
    assert obj.outcomes_col == None
    assert obj.debug == False

def test_gen_rand_periods_get_yml():
    # Test get_yml method
    obj = gen_rand_periods(yml=yml['periods'])
    obj.get_yml()

    assert obj.last_date == pd.Timestamp('2022-12-31')
    assert obj.max_nperiods_back == 40
    assert obj.freq == 'Q'
    assert obj.prob_inactive == 0.05
    assert obj.src_file == 'Resources/81n_per_loc_and_bus.csv'
    assert obj.outcomes_col == 'n_per_each_loc_bus_id'

def test_gen_rand_periods_generate_periods():
    # Test generate_periods method (which are not random)
    obj = gen_rand_periods(yml=yml['periods'])
    periods = obj.generate_periods('2022-12-31', 6, 'Q')
    assert (periods.values == PeriodIndex(['2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3','2022Q4'],
            dtype='period[Q-DEC]')).all()


def test_date_n_periods_back():
    # Case with yml file ('Q')
    obj = gen_rand_periods(yml=yml['periods'])
    assert obj.date_n_periods_back(last_date='2022-12-31', n_back=3, freq='Q') == pd.Timestamp('2022-03-31')
    # Case without yml file and monthly frequency
    obj = gen_rand_periods(freq="M")
    assert obj.date_n_periods_back(last_date = '2022-11-30', n_back=15) == pd.Timestamp('2021-08-30')


def test_get_sample():
    #Testing the generation of sample periods of random length
    # Case1: using yml as input (quarterly periods)
    obj = gen_rand_periods(yml=yml['periods'])
    obj.get_yml()

    assert len(obj.get_sample())>=1
    assert type(obj.get_sample()) == pd.core.indexes.period.PeriodIndex
    assert obj.get_sample().freqstr == 'Q-DEC'

    # Case2: using argumets other than yml as input, using monthly periods
    # Trying case of inactive firm, what it means that the last period ends before the last_date provided.
    obj = gen_rand_periods(last_date='2022-12-31', max_nperiods_back = 40, 
                 freq='M', prob_inactive=1)
    # Object has at least one date
    assert len(obj.get_sample())>=1
    # Object has PeriodIndex type
    assert type(obj.get_sample()) == pd.core.indexes.period.PeriodIndex
    # Object has frequency monthly
    assert obj.get_sample().freqstr == 'M'
    # Last period is before the last_date 
    assert obj.get_sample()[-1] < pd.Period('2022-12-31','M')
    # First period is 40 or less periods before the last period
    sample_index = obj.get_sample()
    end = sample_index[-1] 
    start = sample_index[0]
    num_periods = (end - start).n
    assert num_periods <=40
    

 ##############################################################
 #  TESTING PERIOD GENERATION - gen_random_proportions OBJECT #
 ##############################################################
def test_gen_random_proportions():
        # INIT METHOD - DEFAULT VALUES with YAML fiile
    obj = gen_random_proportions(yml['deducs'])

    assert obj.yml == yml['deducs']
    assert obj.features_p == None
    assert obj.distribution == None
    assert obj.parameters == None
    assert obj.n == 40
    assert obj.seed == 5
    assert obj.normalized == True
    assert obj.export_csv == False
    assert obj.debug == False
    assert obj.include_agg == True
    #assert obj.always_non_neg == None
    assert obj.global_always_non_neg == True

def test_gen_random_proportions_get_yml():
    # Test get_yml method
    obj = gen_random_proportions(yml=yml['deducs'])
    obj.get_yml()

    assert obj.features_p == {
                        'deduc_1': 0.9,
                        'deduc_2': 0.75,
                        'deduc_16': 0.001369,
                        'deduc_17': 0.017539
                        }
    assert obj.distribution == {
                        'deduc_1': 'beta',
                        'deduc_2': 'uniform',
                        'deduc_16': 'uniform',
                        'deduc_17': 'uniform'
        }
    assert obj.parameters == {
                  'deduc_1': {'a': 8, 'b': 8},
                  'deduc_2': {'low': 0, 'high': 1},
                  'deduc_16': {'low': 0, 'high': 1},
                  'deduc_17': {'low': 0, 'high': 1},
                  }
    assert obj.include_agg == False
    assert obj.normalized == True


    
def test_get_sample():
    #Testing the generation of sample periods of random length
    obj = gen_random_proportions(yml=yml['deducs'])
    obj.get_yml()
    deduc_df = obj.get_sample()
    
    assert len(deduc_df) == 40
    assert (isinstance(deduc_df, pd.DataFrame)==True)
    assert (deduc_df.columns == ['deduc_1', 'deduc_2','deduc_16','deduc_17']).all()
    
    # All elements must be proportions
    assert ((deduc_df<=1).sum().sum() == 160)
    

# Function used in the object for normalize proportions
def test_normalize_rows():
    # Case of matrix with multiple columns
    # Include case 0 and non zero
    norm = np.array([[0, 0],[0.2, 0.2]])
    assert (normalize_rows(norm) == np.array([[0,0],[0.5,0.5]])).all()
    
    # Case with matrix of one dimention
    # Case all zero
    assert (normalize_rows(norm[0]) == np.array([0,0])).all()
    
    # Caso non zero
    assert (normalize_rows(norm[1]) == np.array([0.5,0.5])).all()
    

    
#@TODO check if the distributions generated are the ones it supose to be
    
 ####################################################
 #  TESTING MONEY  GENERATION - gen_money_ts OBJECT #
 ####################################################
#@TODO Since this ins gonna change, will leave to the end


 ######################################################################
 #  TESTING OBJECT GENERATION AND POPULATION - ObjectGenerator OBJECT #
 ######################################################################




 ######################################################################
 #  TESTING ONE TIME SERIES GENERATION - gen_ts_records   OBJECT #
 ######################################################################


 ######################################################################
 #  TESTING TIME SERIES DATASET GENERATION - gen_ts_dataset   OBJECT #
 ######################################################################

    
# def test_generate_ts_record():
#     # call the function
#     record_df = generate_ts_record(yml, seed=1, debug=False)
    
#     # test that the output is a DataFrame
#     assert isinstance(record_df, pd.DataFrame)
    
#     # test that the output DataFrame has the correct columns
#     assert set(record_df.columns) == {'business_id', 'income', 'tax_paid', 'effect_tax_rate', 'agg_deduc', 'deduc_1', 'deduc_2', 'deduc_16','deduc_17'}
    
#     # test that the output DataFrame has the correct number of rows
#     # assert len(record_df) == 5
    
#     # test that the output DataFrame has the correct index frequency
#     assert record_df.index.freq == 'Q'
    
