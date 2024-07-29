'''
root/codes/block_definitions/utilities/operators...

Overview:
For the most part just gonna have a basic operator that adds new key/values to a dictionary.
Then each operator will wrap around that one but specify which datatypes it takes.
'''

### packages
import numpy as np


### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.utilities import argument_types



### init dict
operator_dict = {}


def write_dict(hyperparam_dict, **kwargs):
    # kwargs can be treated like a dict and now we can merge them
    # with kwargs able to overwrite shared keywords with hyperparam_dict
    # NOTE this works only for python3.5+
    return {**hyperparam_dict, **kwargs}


def pick_synomym_filter(hyperparam_dict, synonym_filter):
    '''
    until we are ready to switch between files in the experiment_config, we will do nothing in this ftn
    '''
    return write_dict(hyperparam_dict, analyzerWithSynonymFilter=synonym_filter)
operator_dict[pick_synomym_filter] = {"inputs": [dict],
                                      "output": dict,
                                      "args": [argument_types.ArgumentType_SyscoSearch_SynonymFilter]
                                     }


def pick_productdesc_boosts(hyperparam_dict,
                            title_boost,
                            brand_boost,
                            category_boost,
                            line_description_boost):
    return write_dict(hyperparam_dict, **{'titleLocaleBoost': title_boost,
                                          'descriptionLocaleBoost': title_boost, # keep same as title_boost; they're the same value from product data perspective
                                          'brandNameLocaleBoost': brand_boost,
                                          'categoryIntermediateNameLocaleBoost': category_boost,
                                          'lineDescriptionLocaleBoost': line_description_boost})
operator_dict[pick_productdesc_boosts] = {"inputs": [dict],
                                          "output": dict,
                                          "args": [argument_types.ArgumentType_SyscoSearch_ProductDescBoost,
                                                   argument_types.ArgumentType_SyscoSearch_ProductDescBoost,
                                                   argument_types.ArgumentType_SyscoSearch_ProductDescBoost,
                                                   argument_types.ArgumentType_SyscoSearch_ProductDescBoost]
                                          }


def pick_stock_boosts(hyperparam_dict,
                      stock_boost_1,
                      stock_boost_2,
                      stock_boost_3):
    # sort low to high and assign that way
    S_boost, R_boost, D_boost = np.sort([stock_boost_1,stock_boost_2,stock_boost_3])[::-1] # descending
    return write_dict(hyperparam_dict, **{'stockTypeSBoost': int(S_boost), # need to cast as int or else it will stay as np.int64 and will error on json.dump
                                          'stockTypeRBoost': int(R_boost),
                                          'stockTypeDBoost': int(D_boost)})
operator_dict[pick_stock_boosts] = {"inputs": [dict],
                                    "output": dict,
                                    "args": [argument_types.ArgumentType_SyscoSearch_StockBoost,
                                            argument_types.ArgumentType_SyscoSearch_StockBoost,
                                            argument_types.ArgumentType_SyscoSearch_StockBoost]
                                    }


def pick_additional_boosts(hyperparam_dict,
                           isSysco_boost,
                           hasImage_boost,
                           customer_count_factor):
    return write_dict(hyperparam_dict, **{'isSyscoBrandBoost': isSysco_boost,
                                          'imageExistsBoost': hasImage_boost,
                                          'CustomerCountFactorBoost': customer_count_factor})
operator_dict[pick_additional_boosts] = {"inputs": [dict],
                                         "output": dict,
                                         "args": [argument_types.ArgumentType_SyscoSearch_MiscBoost,
                                                  argument_types.ArgumentType_SyscoSearch_MiscBoost,
                                                  argument_types.ArgumentType_SyscoSearch_MiscBoost]
                                         }


def pick_rank_equation(hyperparam_dict,
                       ranking_file):
    '''
    hard to say what this will look like in the future, but for now just going to assume it is like the synonym file stuff...
    we have a prebuilt set of a few files that define how the ranking is done and the hyperparam that get's passed is the
    basename of the files
    
    since this is outside the scope of the typical 'experiment_config' file, we will not actual write anything to the hyperparam dict
    '''
    pass
    return hyperparam_dict
operator_dict[pick_rank_equation] = {"inputs": [dict],
                                     "output": dict,
                                     "args": [argument_types.ArgumentType_SyscoSearch_RankEquationFile]
                                    }