# Test for No-Compliance Random Data Generators Objects and Functions.
# Paola Carvajal Almeida
# pcarvajalalmeida@rsimail.com

from ts_dataset_generator import (
    GenTSDataset, GenTSRecord, 
    GenRandPeriods, GenRandomProportions,
    GenMoneyTS, ObjectGenerator,
    normalize_rows
)
from dstools.datautils.datagen import IdSource
from pandas.testing import assert_frame_equal
import pytest
import pandas as pd
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
from pandas import PeriodIndex
import os

# Sample input for the test
yml = {
    'periods': {
        'last_date': '2022-12-31',
        'max_nperiods_back': 40,
        'freq': 'Q',
        'prob_inactive': 0.05,
        'src_file': 'resources/81n_per_loc_and_bus.csv',
        'outcomes_col': 'n_per_each_loc_bus_id'
    },
    'efftax_aggdeduc_netinc': {
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
        'parameters': {'mean': 250000.0, 'std': 100000.0}
    },
    'deducs': {
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
rng = np.random.default_rng(seed=seed)


def IdSource():
    # Instantiating with rng
    _id = IdSource(rng=rng, length=8)

    # Check the output is an integer
    assert isinstance(_id, np.int64)

    # Check output has eight digits
    assert len(str(_id)) == 8


########################################################
#  TESTING PERIOD GENERATION - GenRandPeriods OBJECT   #
########################################################

def test_GenRandPeriods_initialization():
    # INIT METHOD - DEFAULT VALUES
    obj = GenRandPeriods(yml['periods'], rng=rng)

    assert obj.rng == rng
    assert obj.last_date == pd.Timestamp('2022-12-31')
    assert obj.max_nperiods_back == 40
    assert obj.freq == 'Q'
    assert obj.prob_inactive == 0.1
    assert obj.yml == yml['periods']
    assert obj.src_file is None
    assert obj.export_csv == False
    assert obj.outcomes_col is None
    assert obj.debug == False


def test_GenRandPeriods_get_yml():
    # Test get_yml method
    obj = GenRandPeriods(yml=yml['periods'], rng=rng)
    obj.get_yml()

    assert obj.last_date == pd.Timestamp('2022-12-31')
    assert obj.max_nperiods_back == 40
    assert obj.freq == 'Q'
    assert obj.prob_inactive == 0.05
    
    path_to_src_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'resources','81n_per_loc_and_bus.csv'))
    assert obj.src_file == path_to_src_file, f"src file path incorrectly this: {path_to_src_file}. Must be {obj.src_file} "
    assert obj.outcomes_col == 'n_per_each_loc_bus_id'
    assert obj.rng == rng


def test_GenRandPeriods_generate_periods():
    # Test generate_periods method (which are not random)
    obj = GenRandPeriods(yml=yml['periods'], rng=rng)
    periods = obj.generate_periods('2022-12-31', 6, 'Q')
    assert (periods.values == PeriodIndex(['2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4'],
            dtype='period[Q-DEC]')).all()


def test_date_n_periods_back():
    # Case with yml file ('Q')
    obj = GenRandPeriods(yml=yml['periods'], rng=rng)
    assert obj.date_n_periods_back(
        last_date='2022-12-31', n_back=3, freq='Q') == pd.Timestamp('2022-03-31')

    # Case without yml file and monthly frequency
    obj = GenRandPeriods(freq="M")
    assert obj.date_n_periods_back(
        last_date='2022-11-30', n_back=15) == pd.Timestamp('2021-08-30')


def test_get_sample():
    # Testing the generation of sample periods of random length
    # Case1: using yml as input (quarterly periods)
    obj = GenRandPeriods(yml=yml['periods'], rng=rng)
    obj.get_yml()

    assert len(obj.get_sample()
               ) >= 1, "Expected at least one period, it contains none."
    assert isinstance(obj.get_sample(
    ), pd.core.indexes.period.PeriodIndex), f"Expected an object of type PeriodIndex but it got {type(obj.get_sample())}"
    assert obj.get_sample(
    ).freqstr == 'Q-DEC', f"Expected a quarterly frequency, but obtained {obj.get_sample().freqstr}"

    # Case2: using argumets other than yml as input, using monthly periods
    # Trying case of inactive firm, what it means that the last period ends
    # before the last_date provided.
    obj = GenRandPeriods(rng=rng, last_date='2022-12-31', max_nperiods_back=40,
                         freq='M', prob_inactive=1)

    # Object has at least one date
    assert len(obj.get_sample()) >= 1

    # Object has PeriodIndex type
    assert isinstance(obj.get_sample(), pd.core.indexes.period.PeriodIndex)

    # Object has frequency monthly
    assert obj.get_sample().freqstr == 'M'

    # Last period is before the last_date
    assert obj.get_sample()[-1] < pd.Period('2022-12-31', 'M')

    # First period is 40 or less periods before the last period
    sample_index = obj.get_sample()
    end = sample_index[-1]
    start = sample_index[0]
    num_periods = (end - start).n
    assert num_periods <= 40


##############################################################
#  TESTING PERIOD GENERATION -  GenRandomProportions  OBJECT #
##############################################################

def test_GenRandomProportions():
    # INIT METHOD - DEFAULT VALUES with YAML file
    obj = GenRandomProportions(yml['deducs'], rng=rng)

    assert obj.yml == yml['deducs']
    assert obj.features_p is None
    assert obj.distribution is None
    assert obj.parameters is None
    assert obj.n == 40
    assert obj.normalized == True
    assert obj.export_csv == False
    assert obj.debug == False
    assert obj.include_agg == True
    assert obj.rng == rng

    # assert obj.always_non_neg == None
    assert obj.global_always_non_neg == True


def test_GenRandomProportions_get_yml():
    # Test get_yml method
    obj = GenRandomProportions(yml=yml['deducs'], rng=rng)
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
    # Testing the generation of sample periods of random length
    obj = GenRandomProportions(yml=yml['deducs'], rng=rng)
    obj.get_yml()
    deduc_df = obj.get_sample()

    assert len(deduc_df) == 40
    assert (isinstance(deduc_df, pd.DataFrame) == True)
    assert (deduc_df.columns == ['deduc_1',
            'deduc_2', 'deduc_16', 'deduc_17']).all()

    # All elements must be proportions
    assert ((deduc_df <= 1).sum().sum() == 160)


# Function used in the object for normalize proportions
def test_normalize_rows():
    # Case of matrix with multiple columns
    # Include case 0 and non zero
    norm = np.array([[0, 0], [0.2, 0.2]])
    assert (normalize_rows(norm) == np.array([[0, 0], [0.5, 0.5]])).all()

    # Case with matrix of one dimention
    # Case all zero
    assert (normalize_rows(norm[0]) == np.array([0, 0])).all()

    # Caso non zero
    assert (normalize_rows(norm[1]) == np.array([0.5, 0.5])).all()


####################################################
#  TESTING MONEY  GENERATION -  GenMoneyTS  OBJECT #
####################################################

def test_GenMoneyTS():
    seed = 1
    rng = np.random.default_rng(seed=seed)
    
    income = GenMoneyTS(yml['income'], rng=rng)
    income.get_yml()
    income_df = income.get_sample()

    assert isinstance(
        income_df, pd.DataFrame), f"Expected pd.DataFrame, but got {type(dataset_df)} instead"
    assert income_df.shape[0] > 0, "DataFrame has no elements"
    assert income_df.select_dtypes(include=[
                                   float]).shape == income_df.shape, "DataFrame contains non-float values. Since this is money, it must be float due to cents."
    assert_frame_equal(round(
        income_df,2), income_df), "Numbers are not rounded to 2 decimals. It must since thisin money."


######################################################################
#  TESTING OBJECT GENERATION AND POPULATION - ObjectGenerator OBJECT #
######################################################################

def test_ObjectGenerator():
    # Will test the object generation by generating a distribution of deductions.
    # Dataframe was created
    seed = 1
    rng = np.random.default_rng(seed=seed)
    
    deducs = ObjectGenerator(yml=yml.get('deducs'),
                             rng=rng,
                             class_name='GenRandomProportions',
                             n=40)

    deducs_df = deducs.populate_object()

    assert isinstance(
        deducs_df, pd.DataFrame), f"Expected pd.DataFrame, but got {type(deducs_df)}"

    # Dataframe has content
    assert deducs_df.shape[0] > 0, f"deducs_df Dataframe has no content"


########################################################################
#  TESTING ONE TIME SERIES RECORD GENERATION - GenTSRecord FUNCTION #
########################################################################

def test_GenTSRecord():

    record_gen = GenTSRecord(yml)
    record_df = record_gen.get_record()
    assert isinstance(
        record_df, pd.DataFrame), f"Expected pd.DataFrame, but got {type(record_df)}"
    assert len(
        record_df.columns) == 8, f"Expected 8 columns, but got {len(record_df.columns)}"
    assert (record_df.iloc[:,1:] >= 0).all().all(
    ), "Expected all values to be non-negative"

    # test that the output DataFrame has the correct columns
    assert (record_df.columns.to_list() == [
            'business_id', 'income', 'tax_paid', 'agg_deduc', 'deduc_1', 'deduc_2', 'deduc_16', 'deduc_17'])

    # test that the output DataFrame has the correct index frequency
    assert record_df.index.freq == 'Q', f"Expected frequency to be 'Q', but got: {record_df.index.freq}"


# Testing the transformation
# Data
periods_index = pd.DataFrame({'period':
                              ['2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4']
                              })

id_df = pd.DataFrame({'business_id':
                      [999999, 999999, 999999, 999999, 999999]
                      })

income_df = pd.DataFrame({'income':
                          [0.0, 1.0, 10.0, 100.0, 1000.0]
                          })

efftax_aggdeduc_netinc_df = pd.DataFrame(
    [
        [0.3, 0.33, 0.34],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ],
    columns=['effect_tax_rate', 'agg_deduc', 'net_income']
)

deducs_df = pd.DataFrame(
    [
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.10, 0.20, 0.30, 0.40],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25]
    ],
    columns=['deduc_1', 'deduc_2', 'deduc_16', 'deduc_17']
)

raw_ts_objects = {'periods_index': periods_index,
                  'id_df': id_df,
                  'income_df': income_df,
                  'efftax_aggdeduc_netinc_df': efftax_aggdeduc_netinc_df,
                  'deducs_df': deducs_df
                  }

transformed_df = pd.DataFrame({
    'period': [('2021Q4',), ('2022Q1',), ('2022Q2',), ('2022Q3',), ('2022Q4',)],
    'business_id': [999999, 999999, 999999, 999999, 999999],
    'income': [0.0, 1.0, 10.0, 100.0, 1000.0],
    'tax_paid': [0.0, 0.0, 0.0, 50.0, 500.0],
    'agg_deduc': [0.0, 0.0, 10.0, 0.0, 500.0],
    'deduc_1': [0.0, 0.0, 1.0, 0.0, 125.0],
    'deduc_2': [0.0, 0.0, 2.0, 0.0, 125.0],
    'deduc_16': [0.0, 0.0, 3.0, 0.0, 125.0],
    'deduc_17': [0.0, 0.0, 4.0, 0.0, 125.0]
})
transformed_df.set_index('period', inplace=True)
transformed_df.index.name = None
transformed_df.sort_index(ascending=False, inplace=True)


# Transformation test
def test_transform_objects():
    record_gen = GenTSRecord(yml,rng=rng, raw_ts_objects=raw_ts_objects)
    assert_frame_equal(record_gen.transform_objects(), transformed_df)


######################################################################
#  TESTING TIME SERIES DATASET GENERATION -   GenTSDataset    OBJECT #
######################################################################

def test_GenTSDataset():
    seed = 1
    rng = np.random.default_rng(seed=seed)
    
    dataset_df = GenTSDataset(yml=yml, rng=rng, n_samples=5,
                              export_csv=True).get_dataset()

    # count 5 distinct business ids
    assert len(dataset_df['business_id'].unique()) == 5

    # type is a dataframe
    assert isinstance(
        dataset_df, pd.DataFrame), f"Expected pd.DataFrame, but got {type(dataset_df)}"

    # columns - contents and numbers
    assert (dataset_df.columns.to_list() == [
            'business_id', 'income', 'tax_paid', 'agg_deduc', 'deduc_1', 'deduc_2', 'deduc_16', 'deduc_17'])
    assert len(
        dataset_df.columns) == 8, f"Expected 8 columns, but got {len(dataset_df.columns)}"

    # test that the output DataFrame has the correct index frequency is
    # quarterly
    assert dataset_df.index.freqstr == 'Q-DEC'

    # test that a csv file was generated
    path = './new_dataset.csv'

    def assert_csv_created(path):
        assert os.path.exists(path), f"{path} does not exist"
        assert os.path.isfile(path), f"{path} is not a file"
        assert path.endswith('.csv'), f"{file_path} is not a csv file"
