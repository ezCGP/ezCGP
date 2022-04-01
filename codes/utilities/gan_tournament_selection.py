'''
tournament to rank refiners + discriminators for simgan
'''
import numpy as np
import pandas as pd
import torch

def get_graph_ratings(refiners,
                      discriminators,
                      validation_data,
                      device,
                      starting_rating=1500,
                      starting_rd=350,
                      norm_val=173.7178,
                      n_rounds=3,
                      matches_per_pairing=5,
                      samples_per_match=10,
                      discriminator_win_thresh=0.6):
    '''
    TODO...can we get a Source?
    https://arxiv.org/abs/1808.04888 ?????

    Find the best refiner and discriminator from the list of refiners and discriminators using the Tournament Skill Rating Evaluation.

        Parameters:
            refiners (list(torch.nn)): list of refiners
            discriminators (list(torch.nn)): list of discriminators
            validation_data (simganData): SimGAN dataset
            train_config (dict): dictionary holding information related to training
            starting_rating (float): The rating that players were initialized to
            starting_RD (float): The RD that players were initialized to
            norm_val (float): The normalization value used to convert between phi and RD
            n_rounds(int): Number of rounds for the tournament
            matches_per_pairing(int): The number of matches per refiner/discriminator pairing to determine the overall winner
            samples_per_match(int): The number of samples per match to determine the winner of the match
            discriminator_win_thresh: The accuracy of the discriminator needed for the discriminator to be declared the winner

        Returns:
            A tuple a of Pandas DataFrames...
            A Pandas DataFrame for metadata-ratings where 1 row is for 1 refiner (respectively for discriminator).

    '''
    n_refiners = len(refiners)
    ids = np.arange(n_refiners + len(discriminators))
    refiner_ids = ids[:n_refiners]
    discriminator_ids = ids[n_refiners:]

    ratings = {}
    for id in ids:
        ratings[id] = {'r': starting_rating, 'RD': starting_rd, 'mu': 0, 'phi': starting_rd/norm_val}

    labels_real = torch.zeros(samples_per_match, dtype=torch.float, device=device)
    labels_refined = torch.ones(samples_per_match, dtype=torch.float, device=device)
    all_real = validation_data.real_raw
    all_simulated = validation_data.simulated_raw
    for rnd in range(n_rounds):        
        # instantiate match results
        match_results = {}
        for id in ids:
            match_results[id] = {'opponent_mus': [], 'opponent_phis': [], 'scores': []}

        # Perform matches between each pair (R,D)
        for id_R, R in zip(refiner_ids, refiners):
            for id_D, D in zip(discriminator_ids, discriminators):
                # RODD - ?...why do we need multiple matches? why not just change samples to samples_per_match*matches_per_pairing
                # ...like it's just running data through refiner and discrim. like why not just do that once but with more data?
                for match in range(matches_per_pairing):
                    real_inds = np.random.choice(np.arange(len(all_real)), samples_per_match, replace=False)
                    real = torch.tensor(all_real[real_inds], dtype=torch.float, device=device)
                    sim_inds = np.random.choice(np.arange(len(all_simulated)), samples_per_match, replace=False)
                    simulated = torch.tensor(all_simulated[sim_inds], dtype=torch.float, device=device)
                    refined = R(simulated)
                    
                    # Get discriminator accuracy on real and refined data
                    d_pred_real = D(real)
                    acc_real = calc_acc(d_pred_real, labels_real)
                    d_pred_refined = D(refined)
                    acc_refined = calc_acc(d_pred_refined, labels_refined)

                    # Find the average accuracy of the discriminator
                    avg_acc = (acc_real + acc_refined) / 2.0

                    # Add this match's results to match_results
                    match_results[id_D]['opponent_mus'].append(ratings[id_R]['mu'])
                    match_results[id_R]['opponent_mus'].append(ratings[id_D]['mu'])
                    match_results[id_D]['opponent_phis'].append(ratings[id_R]['phi'])
                    match_results[id_R]['opponent_phis'].append(ratings[id_D]['phi'])
                    if avg_acc >= discriminator_win_thresh: # An accuracy greater than or equal to this threshold is considered a win for the discriminator
                        # A score of 1 is a win
                        match_results[id_D]['scores'].append(1)
                        match_results[id_R]['scores'].append(0)
                    else:
                        match_results[id_D]['scores'].append(0)
                        match_results[id_R]['scores'].append(1)
        
        # Update scores for the refiners and discriminators
        new_ratings = ratings.copy()
        for id in ids:
            results = match_results[id]
            glicko_calculations = calculate_new_glicko_scores(ratings[id]['mu'],
                                                              ratings[id]['phi'],
                                                              np.array(results['opponent_mus']),
                                                              np.array(results['opponent_phis']),
                                                              np.array(results['scores']),
                                                              starting_rating,
                                                              norm_val)   
            new_ratings[id]['mu'], new_ratings[id]['phi'], new_ratings[id]['r'], new_ratings[id]['RD'] = glicko_calculations
        ratings = new_ratings

    # Get refiner and discriminator with best ratings
    ratings_pd = pd.DataFrame(ratings).T
    refiner_ratings = ratings_pd.loc[refiner_ids]
    discriminator_ratings = ratings_pd.loc[discriminator_ids]
    return refiner_ratings, discriminator_ratings


def calc_acc(tensor_output, tensor_labels):
    '''
    Calculate the percent accuracy of the output, using the labels.
    Note that the sigmoid is already calculated as part of the Discriminator Network.
        Parameters:
            tensor_output (torch.Tensor): M tensor output of the discriminator (M samples,) probability of being class '1'
            tensor_labels (torch.Tensor): M tensor true labels for each sample

        Returns:
            acc (float): the probability accuracy of the output vs. the true labels
    '''
    y_pred = torch.round(tensor_output)#.detatch())
    acc = torch.sum(y_pred == tensor_labels.detach()) / len(tensor_labels.detach())
    return acc


def calculate_new_glicko_scores(old_mu, old_phi, opponent_mus, opponent_phis, scores, starting_rating, norm_val):
    '''
    TODO ...Source ????
    http://www.glicko.net/glicko/glicko2.pdf ????
    https://en.wikipedia.org/wiki/Glicko_rating_system ????

    Calculate and return the new glicko values for the player using Glicko2 calculation 
        Parameters:
            old_mu (float): The former mu rating
            old_phi (float): The former phi rating
            opponent_mus (list(float)): The mu ratings of the opponents played
            opponent_phis (list(float)): The phi ratings of the opponents played
            scores (list(inte)): The scores of the games played, 1 indicating a win, 0 indicating a loss
            starting_rating (float): The rating that players were initialized to
            norm_val (float): The normalization value used to convert between phi and RD

        Returns:
            (new_mu, new_phi, new_rating, new_rd) (float, float, float, float): The updated Glicko values for the player
    '''
    g = 1.0 / (1 + 3 * opponent_phis**2 / np.pi**2) ** 0.5 # TODO: explain/figure out what g is
    E = 1.0 / (1 + np.exp(-1 * g * (old_mu - opponent_mus))) # Probability of player winning each match
    v = np.sum(g**2 * E * (1 - E)) ** -1 # Estimated variance of the player's rating based on game outcomes
    delta = v * np.sum(g * (scores - E)) # Estimated improvement in rating
    new_phi = 1 / (1/old_phi**2 + 1/v) ** 0.5
    new_mu = old_mu + new_phi**2 * np.sum(g * (scores - E))
    new_rating = norm_val * new_mu + starting_rating
    new_rd = norm_val * new_phi
    return new_mu, new_phi, new_rating, new_rd
