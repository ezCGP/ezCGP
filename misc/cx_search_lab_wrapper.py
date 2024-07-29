'''
Working off assumption that ezCGP was added as a submodule to cx-search-lab in ./src/scripts/lab
so we need to add that into sys.path for importing the subtasks

wrapper function for all things cx_search_lab in running the experiment ie:
    * lab_judgement_set_create.py
    * lab_judgement_set_analyze.py

'''

### packages
import os, sys
import json
import numpy as np
import pandas as pd
import pdb

### adding ezCGP parent folder into sys.path
# ./ezCGP/misc/__file__ -> ./ezCGP
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from codes.utilities.custom_logging import ezLogging

### adding cx-search-lab parent folder into sys.path
# ./cx-search-lab/src/scripts/lab/ezCGP/misc/__file -> ./cx-search-lab/src/scripts/lab
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import lab_judgement_set_create
import lab_judgement_set_analyze


def load_experiments(experiment_json_filepath):
    ''' #1 Load Experiments
    https://github.com/SyscoCorporation/cx-search-lab/blob/HyperParamOpt--SRC-223_UpdateDAG/src/scripts/lab/lab_judgement_set_create.py#L65C1-L65C13
    in lab_judgement_set_create.py the experiments are loaded via `lab_exeriments_utils.get_all_experiments()` which just reads HARDCODED experiments_configs_v2.json
    so we need to edit that method to take in an arg for the experiment json file, which means we also need lab_judgement_set_create.py to take take in an arg for the
    same file.
    maybe we need to wrap all that into a main() method for the lab_judgement_set_create.py script
    Note that output is parquet file (will overwrite...change?):
        "s3a://cx-search-test-analytics-data/judgement-set-data/cx_staging_judgement_set.parquet"
    '''
    ezLogging.info("Run `lab_judgement_set_create.py` to load in experiments.")

    pass # TODO

    ezLogging.debug("Success - `lab_judgement_set_create.py`")


def run_experiments(experiment_parquet="s3a://cx-search-test-analytics-data/judgement-set-data/cx_staging_judgement_set.parquet"):
    ''' #2 Run Experiments
    https://github.com/SyscoCorporation/cx-search-lab/blob/HyperParamOpt--SRC-223_UpdateDAG/src/scripts/lab/lab_judgement_set_analyze.py#L163
    in lab_judgement_set_analyze.py the experiments are passed in from previous step as `judgement_set_reduced` and then sent to be run with
    `executeRestApi` in the spark query; output of that is stored as `result`.
    the outputs are saved to (will overwrite as well):
        "s3a://cx-search-test-analytics-data/judgement-set-data/cx_analyze_judgement_set.parquet"
    with columns:
        * experiment_id
        * experiment_name
        * result.id
        * result.term_metric_score
    and the metric is defined by the hyperparam file experiment_configs_v2.json in `metric` key:
        "expected_reciprocal_rank": {"maximum_relevance": 10, "k": 20}
        "a search relevance metric that measures how long a user is expected to take to find a relevant document." - Google
        bound by 0 and 1 -> https://notesonai.com/expected+reciprocal+rank 
        I think it is a maximization optimization based off the equation which has a factor of (1/k) where k is the rank of the 
        document, so the higher the rank, the later on in the document it is, and the worse it is, and ERR goes down.
        Higher rank, higher score, maximization. <- TODO
    '''
    ezLogging.info("Run `lab_judgement_set_analyze.py` to run in experiments.")

    pass # TODO

    ezLogging.debug("Success - `lab_judgement_set_analyze.py`")


def get_results(s3_bucket="s3a://cx-search-test-analytics-data/judgement-set-data/cx_analyze_judgement_set.parquet"):
    ''' #3 Get Experiment Results
    reiterated from #2...

    output in 
        "s3a://cx-search-test-analytics-data/judgement-set-data/cx_analyze_judgement_set.parquet"
    with columns:
        * experiment_id
        * experiment_name
        * result.id
        * result.term_metric_score
    
    metric likely a maximization problem
    '''
    ezLogging.info("Request data from S3 bucket")

    pass # TODO
    # get experiment_id and metric
    experiment_results = {} #pd.DataFrame({}) # <- can we turn it into a dict for easy querying later? ...like we don't need to worry about duplicates

    ezLogging.debug("Success - S3 Request")
    
    return experiment_results


def main(experiment_json_filepath, fake=False):
    '''
    TODO do we want it wrapped with try/except?...what will happen if it errors on cloud, how can we log if we can't use try/except and manually debug?
    '''
    if fake:
        ezLogging.warning("`fake` flag is on so we're not actually evaluating but rather passing in fake scores")
        # read in experiment_json_filepath and make some fake scores and read into dict
        experiment_ids = [] # TODO
        fake_scores = np.random.random(len(experiment_ids))
        experiment_results = dict(zip(experiment_ids, list(fake_scores)))
        return experiment_results
    
    try:
        load_experiments(experiment_json_filepath)
        run_experiments()
        get_results()
    except Exception as err:
        ezLogging.error("cx_search_lab_wrapper.main() failed: %s" % err)
        pdb.set_trace()