import os
import math
import time

import json
import numpy as np
import pandas as pd
from io import BytesIO
# from appwrite.id import ID
# from appwrite.services.storage import Storage
import statsmodels.api as sm
from datetime import datetime
# from dotenv import load_dotenv
from scipy.stats import poisson
from appwrite.query import Query
from appwrite.client import Client
from scipy.optimize import minimize
from appwrite.services.databases import Databases


import warnings
warnings.filterwarnings('ignore')


# pip install appwrite
# pip install numpy
# pip install pandas
# pip install requests
# pip install scipy
# pip install statsmodels

# load_dotenv()

def main(context):

    # Replace these with your actual Appwrite credentials
    API_ENDPOINT = 'https://cloud.appwrite.io/v1'
    PROJECT_ID = os.environ['APPWRITE_PROJECT_ID']
    API_KEY = os.environ['APPWRITE_API_KEY']
    DATABASE_ID = os.environ['APPWRITE_DB_ID']
    STORAGE_ID = os.environ['NFD_MODEL_STORAGE']

    # List of collection IDs to retrieve data from
    COLLECTION_IDS = [
        # os.environ['SEASON_MATCHES_NFD20_21'],
        # os.environ['SEASON_MATCHES_NFD21_22'],  
        # os.environ['SEASON_MATCHES_NFD22_23'],
        # os.environ['SEASON_MATCHES_NFD23_24'],
        os.environ['SEASON_MATCHES_NFD24_25'],  
    ]

    # OUTPUT_FILENAME = "nfd_data.xlsx"

    # User-configurable parameter: number of past games to consider for rolling statistics
    # Default is 6, but can be changed as needed
    NUM_PREVIOUS_GAMES = 6
    HALF_LIFE=10
    PRIOR_STRENGTH=0.6,
    UPDATE_INTERVAL=10
    # User-configurable parameter: home advantage factor for expected goals calculation
    # Default is 1.2, but can be changed as needed

    MIN_HOME_ADVANTAGE=0.3
    HOME_ADVANTAGE = 1 + MIN_HOME_ADVANTAGE
    PYTHAGOREAN_EXPONENT=1.83

    # Initialize Appwrite client
    client = Client()
    client.set_endpoint(API_ENDPOINT)
    client.set_project(PROJECT_ID)
    client.set_key(API_KEY)

    # Initialize Databases service
    databases = Databases(client)

    ########################################################

    def save_to_appwrite_storage(dataframe, bucket_id, file_id="nfdStats", client=None, appwrite_endpoint=None, project_id=None, api_key=None):
        """
        Save DataFrame to a single file in Appwrite Storage with consistent name.
        Will replace existing file if it exists.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The DataFrame containing football statistics
        bucket_id : str
            The Appwrite bucket ID where the file will be stored
        file_id : str
            The file ID to use (default: "nfdStats")
        client : appwrite.Client, optional
            An initialized Appwrite client (if not provided, one will be created)
        appwrite_endpoint : str, optional
            The Appwrite endpoint URL (required if client is None)
        project_id : str, optional
            The Appwrite project ID (required if client is None)
        api_key : str, optional
            The Appwrite API key (required if client is None)
            
        Returns:
        --------
        str or None
            The file ID if successful, None otherwise
        """
        import time
        from datetime import datetime
        import pandas as pd
        import numpy as np
        import json
        import os
        import tempfile
        
        # Initialize Appwrite client and services if not provided
        if client is None:
            from appwrite.client import Client
            from appwrite.services.storage import Storage
            from appwrite.exception import AppwriteException
            
            if not appwrite_endpoint or not project_id or not api_key:
                raise ValueError("If client is not provided, you must provide appwrite_endpoint, project_id, and api_key")
            
            # Initialize client with provided credentials
            client = Client()
            client.set_endpoint(appwrite_endpoint)
            client.set_project(project_id)
            client.set_key(api_key)
        
        # Initialize storage service
        from appwrite.services.storage import Storage
        from appwrite.exception import AppwriteException
        from appwrite.input_file import InputFile
        from appwrite.permission import Permission
        storage = Storage(client)
        
        start_time = time.time()
        
        # Make a shallow copy to avoid modifying the original
        df = dataframe.copy()
        context.log(f"Preparing to save {len(df)} records to storage as '{file_id}'")
        
        # Check for and process specific columns
        if 'date' in df.columns:
            context.log("Column 'date' found in the dataframe")
            context.log("Formatting date column...")
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Handle column renames if needed
        if 'full_time_result' in df.columns and 'ft_result' not in df.columns:
            context.log("Column 'full_time_result' found in the dataframe")
            context.log("Renamed 'full_time_result' to 'ft_result'")
            df = df.rename(columns={'full_time_result': 'ft_result'})
        
        # Reorder columns if needed
        if all(col in df.columns for col in ['home_goals', 'away_goals', 'goal_diff']):
            # Get the current column order
            cols = df.columns.tolist()
            # Find the position of 'away_goals'
            away_goals_pos = cols.index('away_goals')
            # Remove 'goal_diff' from its current position
            if 'goal_diff' in cols:
                cols.remove('goal_diff')
                # Insert 'goal_diff' after 'away_goals'
                cols.insert(away_goals_pos + 1, 'goal_diff')
                # Reorder the dataframe
                df = df[cols]
                context.log("Moved 'goal_diff' column to come after 'away_goals'")
        
        # Handle boolean columns
        if 'rolling_stats_valid' in df.columns:
            df['rolling_stats_valid'] = df['rolling_stats_valid'].fillna(False).astype(bool)
        
        # Process numeric columns to ensure proper JSON serialization
        integer_columns = [
            'homeID', 'awayID', 'winningTeam', 'result', 
            'home_goals', 'away_goals', 'goal_diff',
            'home_team_goals_scored_total', 
            'home_team_goals_conceded_total',
            'away_team_goals_scored_total', 
            'away_team_goals_conceded_total'
        ]
        
        float_columns = [
            'odds_ft_1', 'odds_ft_x', 'odds_ft_2',
            'home_xg_odds', 'away_xg_odds',
            'home_team_goals_scored_average', 'home_team_goals_conceded_average',
            'away_team_goals_scored_average', 'away_team_goals_conceded_average',
            'home_xg', 'away_xg', 'home_xg_elo', 'away_xg_elo',
            'home_xg_dc', 'away_xg_dc', 'home_xg_bt', 'away_xg_bt',
            'home_xg_pyth', 'away_xg_pyth', 'home_xg_bayesian', 'away_xg_bayesian',
            'home_xg_twr', 'away_xg_twr'
        ]
        
        # Clean up data for serialization
        for col in integer_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        for col in float_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype(float)
        
        # Add metadata as part of the JSON
        metadata = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "record_count": len(df)
        }
        
        # Create a JSON object with metadata and data
        json_data = {
            "metadata": metadata,
            "data": df.to_dict(orient='records')
        }
        
        # Save JSON to a temporary file (Appwrite expects a file path, not a BytesIO object)
        # Create a temporary file to store the JSON data
        temp_fd, temp_path = tempfile.mkstemp(suffix='.json')
        try:
            with os.fdopen(temp_fd, 'w') as tmp:
                json.dump(json_data, tmp, default=str)
            
            context.log(f"Saved data to temporary file: {temp_path}")
            
            # First, check if file exists and delete it
            try:
                # Try to get file info to check existence
                storage.get_file(bucket_id=bucket_id, file_id=file_id)
                
                # If we reach here, file exists, so delete it
                context.log(f"Existing file '{file_id}' found. Deleting...")
                storage.delete_file(bucket_id=bucket_id, file_id=file_id)
                context.log(f"Existing file deleted successfully.")
            except AppwriteException as e:
                # File doesn't exist, which is fine
                if "not found" in str(e).lower():
                    context.log(f"No existing file with ID '{file_id}' found. Creating new file.")
                else:
                    context.log(f"Warning when checking existing file: {str(e)}")
            
            # Try two approaches for permissions
            try:
                # First approach - without specifying permissions (use bucket defaults)
                result = storage.create_file(
                    bucket_id=bucket_id,
                    file_id=file_id,
                    file=InputFile.from_path(temp_path)
                    # No permissions parameter
                )
            except AppwriteException as e1:
                # If that fails, try with explicit permissions
                if "permissions" in str(e1).lower():
                    try:
                        # Second approach - specify explicit 'read' permission
                        result = storage.create_file(
                            bucket_id=bucket_id,
                            file_id=file_id,
                            file=InputFile.from_path(temp_path),
                            permissions=['read']  # Allow read permission only
                        )
                    except AppwriteException as e2:
                        # If that also fails, context.log both errors and give up
                        context.log(f"Failed with default permissions: {str(e1)}")
                        context.log(f"Failed with explicit 'read' permission: {str(e2)}")
                        raise e2
                else:
                    # Not a permissions error, rethrow
                    raise e1
            
            total_time = time.time() - start_time
            context.log(f"✅ Successfully saved {len(df)} records to storage in {total_time:.2f}s")
            context.log(f"File ID: {result['$id']}")
            
            return result['$id']
            
        except Exception as e:
            total_time = time.time() - start_time
            context.log(f"❌ Error uploading to storage after {total_time:.2f}s: {str(e)}")
            return None
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                context.log(f"Deleted temporary file: {temp_path}")
        
    ########################################################

    # Function to convert odds to raw probabilities
    def odds_to_raw_probs(home_odds, draw_odds, away_odds):
        home_prob = 1 / home_odds
        draw_prob = 1 / draw_odds
        away_prob = 1 / away_odds
        return home_prob, draw_prob, away_prob

    # Function to normalize probabilities (account for overround)
    def normalize_probs(home_prob, draw_prob, away_prob):
        total = home_prob + draw_prob + away_prob
        return home_prob/total, draw_prob/total, away_prob/total

    # Function to calculate match outcome probabilities from Poisson parameters
    def poisson_match_probs(params):
        home_xg, away_xg = params
        
        # Calculate probabilities for different scorelines (0-0, 1-0, 0-1, etc.)
        max_goals = 10  # Consider up to 10 goals for each team
        home_probs = np.exp(-home_xg) * np.power(home_xg, np.arange(max_goals)) / np.array([math.factorial(i) for i in range(max_goals)])
        away_probs = np.exp(-away_xg) * np.power(away_xg, np.arange(max_goals)) / np.array([math.factorial(i) for i in range(max_goals)])
        
        # Calculate match outcome probabilities
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        for i in range(max_goals):
            for j in range(max_goals):
                p = home_probs[i] * away_probs[j]
                if i > j:
                    home_win_prob += p
                elif i == j:
                    draw_prob += p
                else:
                    away_win_prob += p
        
        return home_win_prob, draw_prob, away_win_prob

    # Function to minimize (difference between bookmaker probabilities and Poisson probabilities)
    def objective_function(params, target_probs):
        home_win_prob, draw_prob, away_win_prob = poisson_match_probs(params)
        target_home_prob, target_draw_prob, target_away_prob = target_probs
        
        # Sum of squared differences
        return (
            (home_win_prob - target_home_prob)**2 + 
            (draw_prob - target_draw_prob)**2 + 
            (away_win_prob - target_away_prob)**2
        )

    # Function to solve for xG values
    def solve_for_xg(home_odds, draw_odds, away_odds):
        # Convert odds to normalized probabilities
        raw_probs = odds_to_raw_probs(home_odds, draw_odds, away_odds)
        target_probs = normalize_probs(*raw_probs)
        
        # Initial guess for xG values (reasonable starting point)
        initial_guess = [1.5, 1.0]
        
        # Bounds to ensure positive xG values
        bounds = [(0.01, 5), (0.01, 5)]
        
        # Solve the optimization problem
        result = minimize(
            objective_function, 
            initial_guess, 
            args=(target_probs,), 
            method='L-BFGS-B', 
            bounds=bounds
        )
        
        # Return the optimized parameters (home_xg, away_xg)
        return result.x

    # Apply to the DataFrame
    def add_xg_columns(df):
        # Initialize empty lists for home and away xG
        home_xg_list = []
        away_xg_list = []
        
        # Process each row
        for _, row in df.iterrows():
            try:
                home_odds = row['odds_ft_1']
                draw_odds = row['odds_ft_x']
                away_odds = row['odds_ft_2']
                
                # Check if odds are valid
                if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds) or home_odds <= 0 or draw_odds <= 0 or away_odds <= 0:
                    # Set to 0 instead of np.nan for missing odds
                    home_xg_list.append(0.0)
                    away_xg_list.append(0.0)
                else:
                    home_xg, away_xg = solve_for_xg(home_odds, draw_odds, away_odds)
                    home_xg_list.append(round(home_xg,2))
                    away_xg_list.append(round(away_xg,2))
            except Exception as e:
                context.log(f"Error processing row: {e}")
                # Set to 0 instead of np.nan for errors too
                home_xg_list.append(0.0)
                away_xg_list.append(0.0)
        
        # Add columns to DataFrame
        df['home_xg_odds'] = home_xg_list
        df['away_xg_odds'] = away_xg_list
        
        return df

    ########################################################

    # Function to calculate expected goals using Bayesian parameters
    def calculate_expected_goals(home_team, away_team, attack_params, defense_params, home_advantage=1.2):
        """
        Calculate expected goals for a match using Bayesian parameters.
        
        Parameters:
        -----------
        home_team, away_team : str
            Team names
        attack_params, defense_params : dict
            Dictionaries containing attack and defense parameters for all teams
        home_advantage : float
            Home advantage factor (default: 1.2)
        
        Returns:
        --------
        tuple
            (expected_home_goals, expected_away_goals)
        """
        # Get parameters - default to middle value (1.5) if not available
        home_attack_alpha, _ = attack_params.get(home_team, (1.5, 1.0))
        home_defense_alpha, _ = defense_params.get(home_team, (1.5, 1.0))
        away_attack_alpha, _ = attack_params.get(away_team, (1.5, 1.0))
        away_defense_alpha, _ = defense_params.get(away_team, (1.5, 1.0))
        
        # For defense, higher values mean better defense (fewer goals conceded)
        # We need to convert defense rating to a factor that reduces expected goals
        home_defense_factor = (3.0 - home_defense_alpha) / 3.0  # Transform to a 0-1 scale
        away_defense_factor = (3.0 - away_defense_alpha) / 3.0  # Transform to a 0-1 scale
        
        # Calculate expected goals based on team's attack strength against opponent's defense
        # Home team expected goals = home team attack * away team defense factor * home advantage
        # Away team expected goals = away team attack * home team defense factor
        expected_home_goals = home_attack_alpha * away_defense_factor * home_advantage
        expected_away_goals = away_attack_alpha * home_defense_factor
        
        # Ensure expected goals are within the 0-3 range
        expected_home_goals = min(max(expected_home_goals, 0.0), 3.0)
        expected_away_goals = min(max(expected_away_goals, 0.0), 3.0)
        
        return expected_home_goals, expected_away_goals

    ########################################################

    # Function to calculate rolling statistics for teams
    def add_team_rolling_stats(df, num_previous_games=6, home_advantage=1.2):
        """
        Add rolling statistics for each team based on their past n games.
        Statistics are calculated independently for each season.
        
        Args:
            df: DataFrame containing match data
            num_previous_games: Number of previous games to consider for rolling stats
            home_advantage: Home advantage factor for expected goals calculation
            
        Returns:
            DataFrame with additional columns for team statistics
        """
        context.log(f"Calculating rolling statistics based on the last {num_previous_games} games...")
        
        # Ensure the DataFrame is sorted by date
        df = df.sort_values(by=['season', 'match_date'])
        
        # Initialize new columns for rolling statistics
        stats_columns = [
            'home_team_goals_scored_total', 'home_team_goals_conceded_total',
            'away_team_goals_scored_total', 'away_team_goals_conceded_total',
            'home_team_goals_scored_average', 'home_team_goals_conceded_average',
            'away_team_goals_scored_average', 'away_team_goals_conceded_average',
            'home_xg', 'away_xg'
        ]
        
        for col in stats_columns:
            df[col] = 0.0
        
        # Process each season separately
        seasons = df['season'].unique()
        
        for season in seasons:
            context.log(f"Processing season: {season}")
            
            # Filter data for current season
            season_df = df[df['season'] == season].copy()
            
            # Get all unique teams in this season
            home_teams = season_df['homeID'].unique()
            away_teams = season_df['awayID'].unique()
            all_teams = np.union1d(home_teams, away_teams)
            
            # Initialize dictionary to track team performance
            team_performance = {team_id: [] for team_id in all_teams}
            
            # Initialize dictionaries for attack and defense parameters
            attack_params = {}
            defense_params = {}
            
            # Build team performance history
            for idx, row in season_df.iterrows():
                # For home team, add match to history
                home_team_id = row['homeID']
                away_team_id = row['awayID']
                
                # Skip if match is not complete
                if row['status'] != 'complete':
                    continue
                    
                # Add match to home team history
                team_performance[home_team_id].append({
                    'match_date': row['match_date'],
                    'is_home': True,
                    'goals_scored': row['homeGoalCount'],
                    'goals_conceded': row['awayGoalCount']
                })
                
                # Add match to away team history
                team_performance[away_team_id].append({
                    'match_date': row['match_date'],
                    'is_home': False,
                    'goals_scored': row['awayGoalCount'],
                    'goals_conceded': row['homeGoalCount']
                })
            
            # Update attack and defense parameters based on team performance
            for team_id, matches in team_performance.items():
                if not matches:
                    continue
                    
                # Use up to the most recent num_previous_games matches
                recent_matches = matches[-num_previous_games:] if len(matches) > num_previous_games else matches
                
                # Calculate average goals scored and conceded
                goals_scored = [match['goals_scored'] for match in recent_matches]
                goals_conceded = [match['goals_conceded'] for match in recent_matches]
                
                avg_goals_scored = sum(goals_scored) / len(recent_matches) if recent_matches else 1.5
                avg_goals_conceded = sum(goals_conceded) / len(recent_matches) if recent_matches else 1.5
                
                # Map averages to parameters in 0-3 range
                # For attack, higher is better - this represents the team's goal-scoring ability
                attack_alpha = min(max(avg_goals_scored, 0.0), 3.0)
                
                # For defense, lower conceded is better, so we use an inverse scale
                # Higher defense_alpha means better defense (fewer goals conceded)
                defense_alpha = min(max(3.0 - avg_goals_conceded, 0.0), 3.0)
                
                # Store parameters
                attack_params[team_id] = (attack_alpha, 1.0)  # using fixed beta for simplicity
                defense_params[team_id] = (defense_alpha, 1.0)  # using fixed beta for simplicity
            
            # Now calculate rolling statistics for each match
            for idx, row in season_df.iterrows():
                match_date = row['match_date']
                home_team_id = row['homeID']
                away_team_id = row['awayID']
                
                # Calculate home team stats
                home_team_history = team_performance[home_team_id]
                # Filter history to only include matches before current match
                previous_home_matches = [
                    match for match in home_team_history
                    if match['match_date'] < match_date
                ]

                # Get the number of previous matches for home team
                num_home_previous_all = len(previous_home_matches)
                
                # Use the specified number of previous games or all available if less
                previous_home_matches = previous_home_matches[-num_previous_games:] if previous_home_matches else []
                num_home_previous = len(previous_home_matches)
                
                # Calculate home team totals
                home_goals_scored = sum(match['goals_scored'] for match in previous_home_matches)
                home_goals_conceded = sum(match['goals_conceded'] for match in previous_home_matches)
                
                # Calculate home team averages
                home_goals_scored_avg = home_goals_scored / num_home_previous if num_home_previous > 0 else 0
                home_goals_conceded_avg = home_goals_conceded / num_home_previous if num_home_previous > 0 else 0
                
                # Calculate away team stats
                away_team_history = team_performance[away_team_id]
                # Filter history to only include matches before current match
                previous_away_matches = [
                    match for match in away_team_history
                    if match['match_date'] < match_date
                ]

                # Get the number of previous matches for away team
                num_away_previous_all = len(previous_away_matches)
                
                # Use the specified number of previous games or all available if less
                previous_away_matches = previous_away_matches[-num_previous_games:] if previous_away_matches else []
                num_away_previous = len(previous_away_matches)
                
                # Calculate away team totals
                away_goals_scored = sum(match['goals_scored'] for match in previous_away_matches)
                away_goals_conceded = sum(match['goals_conceded'] for match in previous_away_matches)
                
                # Calculate away team averages
                away_goals_scored_avg = away_goals_scored / num_away_previous if num_away_previous > 0 else 0
                away_goals_conceded_avg = away_goals_conceded / num_away_previous if num_away_previous > 0 else 0
                
                # Calculate expected goals using Bayesian parameters
                expected_home_goals, expected_away_goals = calculate_expected_goals(
                    home_team_id, 
                    away_team_id, 
                    attack_params, 
                    defense_params, 
                    home_advantage
                )
                
                # Update the DataFrame with calculated values
                df.loc[idx, 'home_team_goals_scored_total'] = int(home_goals_scored)
                df.loc[idx, 'home_team_goals_conceded_total'] = int(home_goals_conceded)
                df.loc[idx, 'away_team_goals_scored_total'] = int(away_goals_scored)
                df.loc[idx, 'away_team_goals_conceded_total'] = int(away_goals_conceded)
                df.loc[idx, 'home_team_goals_scored_average'] = round(home_goals_scored_avg, 2)
                df.loc[idx, 'home_team_goals_conceded_average'] = round(home_goals_conceded_avg, 2)
                df.loc[idx, 'away_team_goals_scored_average'] = round(away_goals_scored_avg, 2)
                df.loc[idx, 'away_team_goals_conceded_average'] = round(away_goals_conceded_avg, 2)
                df.loc[idx, 'home_xg'] = round(expected_home_goals, 2)
                df.loc[idx, 'away_xg'] = round(expected_away_goals, 2)

                # Set the rolling_stats_valid flag
                # Flag is 1 only if both teams have played at least num_previous_games games in this season
                if num_home_previous_all >= num_previous_games and num_away_previous_all >= num_previous_games:
                    df.loc[idx, 'rolling_stats_valid'] = 1
                else:
                    df.loc[idx, 'rolling_stats_valid'] = 0
        
        return df

    ########################################################

    # Function to retrieve all documents from a collection
    def get_collection_documents(collection_id):
        all_documents = []
        limit = 400
        offset = 0
        
        while True:
            try:
                # Set up query options for offset-based pagination
                query_options = [
                    Query.limit(limit),
                    Query.offset(offset)
                ]
                
                # Get a batch of documents
                response = databases.list_documents(
                    database_id=DATABASE_ID,
                    collection_id=collection_id,
                    queries=query_options
                )
                
                documents = response['documents']
                
                # If no documents were returned, we've reached the end
                if not documents:
                    break
                    
                # Add documents to our list
                all_documents.extend(documents)
                
                context.log(f"Retrieved {len(all_documents)} documents from collection {collection_id} so far...")
                
                # If we got fewer documents than the limit, we've reached the end
                if len(documents) < limit:
                    break
                    
                # Update offset for the next page
                offset += limit
                
            except Exception as e:
                context.log(f"Error retrieving documents from collection {collection_id}: {str(e)}")
                break
        
        return all_documents

    # Create an empty list to store dataframes from each collection
    all_dataframes = []

    # Process each collection and create a dataframe
    for collection_id in COLLECTION_IDS:
        if collection_id:  # Skip if collection ID is None or empty
            context.log(f"\nProcessing collection: {collection_id}")
            documents = get_collection_documents(collection_id)
            
            if documents:
                # Create a DataFrame for this collection
                df = pd.DataFrame(documents)
                
                # Add a column to identify which collection this data came from
                df['source_collection'] = collection_id
                
                # Add this dataframe to our list
                all_dataframes.append(df)
                context.log(f"Added {len(df)} rows from collection {collection_id}")
            else:
                context.log(f"No documents found in collection {collection_id}")

    # Combine all dataframes into one
    if all_dataframes:
        # Concatenate all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        context.log(f"\nCombined DataFrame has {len(combined_df)} rows")
        
        # Replace -1 with 0 only in numeric columns
        numeric_columns = combined_df.select_dtypes(include=['number']).columns
        combined_df[numeric_columns] = combined_df[numeric_columns].replace(-1, 0)
        
        # Create a copy of the DataFrame to defragment it
        combined_df = combined_df.copy()
        
        # Create a date column if date_unix exists
        if 'date_unix' in combined_df.columns:
            # Process all dates at once using vectorized operations
            combined_df['date_unix'] = pd.to_numeric(combined_df['date_unix'], errors='coerce')
            
            # Create a datetime column for sorting
            combined_df['datetime_temp'] = pd.to_datetime(combined_df['date_unix'], unit='s')
            
            # Create a formatted date string column
            combined_df['match_date'] = combined_df['datetime_temp'].dt.strftime('%Y-%m-%d')
            
            # Sort by the datetime column from earliest to latest
            combined_df = combined_df.sort_values(by='datetime_temp')
            
            # Drop the temporary datetime column (optional)
            combined_df = combined_df.drop(columns=['datetime_temp'])
            
        # Create the 'winningTeamName' column using a lambda function
        combined_df['winningTeamName'] = combined_df.apply(
            lambda row: row['home_name'] if row['winningTeam'] == row['homeID'] 
                        else row['away_name'] if row['winningTeam'] == row['awayID'] 
                        else 'Draw' if row['winningTeam'] == 0 
                        else None,  # This handles any unexpected values
            axis=1  
        )   

        # Filter for essential columns first
        combined_df = combined_df[['match_date', 'season', 'status', 'homeID', 'home_name', 'awayID', 'away_name', 'winningTeam', 'winningTeamName',
                                'homeGoalCount','awayGoalCount', 'odds_ft_1', 'odds_ft_x', 'odds_ft_2']]
        
        # Apply xG calculations
        combined_df = add_xg_columns(combined_df)
        
        # Add the new rolling statistics including expected goals with the specified home advantage
        combined_df = add_team_rolling_stats(combined_df, num_previous_games=NUM_PREVIOUS_GAMES, home_advantage=HOME_ADVANTAGE)

        # Save to Excel
        # combined_df.to_excel(OUTPUT_FILENAME, index=False)
        # context.log(f"\nData saved to {OUTPUT_FILENAME}")
        
        # context.log first few rows
        # context.log("\nFirst 5 rows of the combined data:")
        # context.log(combined_df.head())
    else:
        context.log("No data found in any of the collections")

    def formatDataframe(df):



            
        # Check if 'date' column exists
        if 'date' in df.columns:
            context.log("Column 'date' found in the dataframe")
        else:
            context.log("Column 'date' NOT found. Available columns:", df.columns.tolist())
            
            # Check for case-insensitive matches
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                context.log(f"Found possible date columns: {date_cols}")
                # Use the first match
                df = df.rename(columns={date_cols[0]: 'date'})
            else:
                # If no date column is found, create a column to avoid errors
                context.log("No date column found. Using a placeholder.")
                df['date'] = pd.NaT
                
        # Now we can safely format the date column
        context.log("Formatting date column...")
        df['date'] = pd.to_datetime(df['date']).dt.date

        # 2. Add new column 'result' that maps full_time_result values
        # First check if full_time_result column exists
        if 'full_time_result' in df.columns:
            context.log("Column 'full_time_result' found in the dataframe")
            result_mapping = {'H': 1, 'D': 0, 'A': 2}
            df['result'] = df['full_time_result'].map(result_mapping)
        else:
            context.log("Column 'full_time_result' NOT found. Available columns:", df.columns.tolist())
            # Check for alternative column names
            result_cols = [col for col in df.columns if 'result' in col.lower() and 'full' in col.lower()]
            if result_cols:
                context.log(f"Found possible result columns: {result_cols}")
                # Use the first match
                result_col = result_cols[0]
                df = df.rename(columns={result_col: 'full_time_result'})
                result_mapping = {'H': 1, 'D': 0, 'A': 2}
                df['result'] = df['full_time_result'].map(result_mapping)
            else:
                context.log("No suitable result column found. Creating a placeholder.")
                df['full_time_result'] = None
                df['result'] = None

        # 3. Rename full_time_result to ft_result
        if 'full_time_result' in df.columns:
            df = df.rename(columns={'full_time_result': 'ft_result'})
            context.log("Renamed 'full_time_result' to 'ft_result'")
        else:
            context.log("Cannot rename 'full_time_result' as it does not exist in the dataframe")

        # 4. Move result and ft_result columns to come after winningTeamName column
        # First, get the index of 'winningTeamName' column
        winning_team_idx = df.columns.get_loc('winningTeamName')

        # Extract the columns we want to move
        result_col = df['result']
        ft_result_col = df['ft_result']

        # Drop those columns from the DataFrame
        df = df.drop(['result', 'ft_result'], axis=1)

        # Insert them after 'winningTeamName' column
        df.insert(winning_team_idx + 1, 'result', result_col)
        df.insert(winning_team_idx + 2, 'ft_result', ft_result_col)

        # 5. Move goal_diff column to come after away_goals
        if 'goal_diff' in df.columns and 'away_goals' in df.columns:
            # Get the index of away_goals column
            away_goals_idx = df.columns.get_loc('away_goals')
            
            # Extract the goal_diff column
            goal_diff_col = df['goal_diff']
            
            # Drop the column
            df = df.drop('goal_diff', axis=1)
            
            # Insert it after away_goals
            df.insert(away_goals_idx + 1, 'goal_diff', goal_diff_col)
            context.log("Moved 'goal_diff' column to come after 'away_goals'")
        else:
            context.log("'goal_diff' or 'away_goals' column not found - cannot move goal_diff")

        return df

    def load_and_process_data(df):
        """
        Load match data from Excel and prepare for analysis.
        With specific handling for various column naming conventions.
        
        Parameters:
        file_path - Path to the Excel file
        
        Returns:
        Processed DataFrame
        """
        try:
            # Load the Excel file
            # df = pd.read_excel(file_path)
            context.log(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns.")
            context.log(f"Columns found: {df.columns.tolist()}")
            
            # Specifically map the known column names in your file
            column_mapping = {}
            
            # Add the specific mappings you identified
            if 'homeGoalCount' in df.columns:
                column_mapping['homeGoalCount'] = 'home_goals'
            
            if 'awayGoalCount' in df.columns:
                column_mapping['awayGoalCount'] = 'away_goals'
            
            # Additional mappings for other columns we need
            column_mappings = {
                'home_team': ['hometeam', 'home_team', 'homename', 'home_name', 'home team', 'home', 'home_id', 'homeid', 'hometeam'],
                'away_team': ['awayteam', 'away_team', 'awayname', 'away_name', 'away team', 'away', 'away_id', 'awayid', 'awayteam'],
                'date': ['date', 'match_date', 'gamedate', 'game_date', 'datetime', 'date_time'],
                'season': ['season', 'seasonid', 'season_id', 'year', 'competition_year'],
                'full_time_result': ['ftr', 'full_time_result', 'result', 'outcome', 'match_result']
            }
            
            # Map other necessary columns
            for target_col, possible_names in column_mappings.items():
                # Skip if we already have this column
                if target_col in df.columns or target_col in column_mapping.values():
                    continue
                    
                # Try various matching approaches
                for possible_name in possible_names:
                    if possible_name in df.columns:
                        column_mapping[possible_name] = target_col
                        break
                
                # If still not found, try case-insensitive matching
                if target_col not in column_mapping.values():
                    for col in df.columns:
                        if col.lower() in possible_names:
                            column_mapping[col] = target_col
                            break
            
            # Apply the mappings found
            if column_mapping:
                context.log(f"\nRenaming columns: {column_mapping}")
                df = df.rename(columns=column_mapping)
            
            # Check for still missing required columns
            required_cols = ['home_team', 'away_team', 'home_goals', 'away_goals']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                # Special handling for team columns
                if 'home_team' in missing_cols:
                    home_team_candidates = [col for col in df.columns if 'home' in col.lower() and ('team' in col.lower() or 'name' in col.lower())]
                    if home_team_candidates:
                        df['home_team'] = df[home_team_candidates[0]]
                        context.log(f"Using '{home_team_candidates[0]}' as home_team")
                        missing_cols.remove('home_team')
                    elif 'homeID' in df.columns:
                        df['home_team'] = df['homeID'].astype(str)
                        context.log("Using homeID as home_team")
                        missing_cols.remove('home_team')
                
                if 'away_team' in missing_cols:
                    away_team_candidates = [col for col in df.columns if 'away' in col.lower() and ('team' in col.lower() or 'name' in col.lower())]
                    if away_team_candidates:
                        df['away_team'] = df[away_team_candidates[0]]
                        context.log(f"Using '{away_team_candidates[0]}' as away_team")
                        missing_cols.remove('away_team')
                    elif 'awayID' in df.columns:
                        df['away_team'] = df['awayID'].astype(str)
                        context.log("Using awayID as away_team")
                        missing_cols.remove('away_team')
                
                # If still missing required columns, raise error
                if missing_cols:
                    context.log("\nStill missing required columns. Here are all available columns:")
                    for i, col in enumerate(df.columns):
                        context.log(f"{i}: {col}")
                    
                    raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Try to create full_time_result if it doesn't exist
            if 'full_time_result' not in df.columns and 'home_goals' in df.columns and 'away_goals' in df.columns:
                df['full_time_result'] = np.where(df['home_goals'] > df['away_goals'], 'H',
                                    np.where(df['home_goals'] < df['away_goals'], 'A', 'D'))
                context.log("Created 'full_time_result' from goals comparison")
            
            # Ensure date column is datetime
            if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Calculate goal difference
            df['goal_diff'] = df['home_goals'] - df['away_goals']
            
            # Try to create season if it doesn't exist
            if 'season' not in df.columns and 'date' in df.columns:
                try:
                    if pd.api.types.is_datetime64_dtype(df['date']):
                        # Extract season (assuming seasons span calendar years, e.g., 2020-2021)
                        df['season'] = df['date'].dt.year.astype(str) + '-' + (df['date'].dt.year + 1).astype(str)
                        context.log("Created 'season' column from 'date'")
                except Exception as e:
                    context.log(f"Could not extract season from date: {e}")
            
            context.log("\nProcessed data summary:")
            context.log(f"Rows: {len(df)}")
            if 'home_team' in df.columns:
                unique_teams = set(df['home_team'].unique()).union(set(df['away_team'].unique()))
                context.log(f"Unique teams: {len(unique_teams)}")
            if 'season' in df.columns:
                context.log(f"Seasons: {df['season'].unique()}")
            
            return df
        
        except Exception as e:
            context.log(f"Error loading data: {e}")
            context.log("\nTrying to display file contents for debugging:")
            try:
                # Try to just read the raw Excel without processing
                # raw_df = pd.read_excel(file_path)
                raw_df=df
                context.log(f"Raw columns: {raw_df.columns.tolist()}")
                context.log(raw_df.head(3))
            except Exception as sub_e:
                context.log(f"Could not read raw file: {sub_e}")
            return None

    clean_dataframe = load_and_process_data(combined_df)




    ###################################################################################################
    #################################     ELO MODEL    ################################################



    # Function to calculate match outcome probabilities from Poisson parameters
    def poisson_match_probs(home_xg, away_xg):
        """
        Calculate match outcome probabilities using Poisson distribution.
        
        Parameters:
        -----------
        home_xg : float
            Expected goals for home team
        away_xg : float
            Expected goals for away team
            
        Returns:
        --------
        tuple (home_win_prob, draw_prob, away_win_prob)
            Probabilities for each match outcome
        """
        # Calculate probabilities for different scorelines (0-0, 1-0, 0-1, etc.)
        max_goals = 10  # Consider up to 10 goals for each team
        home_probs = np.exp(-home_xg) * np.power(home_xg, np.arange(max_goals)) / np.array([math.factorial(i) for i in range(max_goals)])
        away_probs = np.exp(-away_xg) * np.power(away_xg, np.arange(max_goals)) / np.array([math.factorial(i) for i in range(max_goals)])
        
        # Calculate match outcome probabilities
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        for i in range(max_goals):
            for j in range(max_goals):
                p = home_probs[i] * away_probs[j]
                if i > j:
                    home_win_prob += p
                elif i == j:
                    draw_prob += p
                else:
                    away_win_prob += p
        
        return home_win_prob, draw_prob, away_win_prob

    # Function to solve for xG values that match target probabilities
    def solve_for_xg(target_probs, league_avg_home_xg=1.35, league_avg_away_xg=1.1):
        """
        Solve for xG values that would produce the target match probabilities.
        
        Parameters:
        -----------
        target_probs : tuple
            Target probabilities (home_win, draw, away_win)
        league_avg_home_xg : float
            League average for home team expected goals
        league_avg_away_xg : float
            League average for away team expected goals
            
        Returns:
        --------
        tuple (home_xg, away_xg)
            Expected goals values that produce probabilities closest to target
        """
        target_home_prob, target_draw_prob, target_away_prob = target_probs
        
        # Function to minimize (difference between Poisson probabilities and target probabilities)
        def objective_function(params):
            home_xg, away_xg = params
            home_win_prob, draw_prob, away_win_prob = poisson_match_probs(home_xg, away_xg)
            
            # Sum of squared differences
            return (
                (home_win_prob - target_home_prob)**2 + 
                (draw_prob - target_draw_prob)**2 + 
                (away_win_prob - target_away_prob)**2
            )
        
        # Initial guess for xG values (reasonable starting points based on league averages)
        initial_guess = [league_avg_home_xg, league_avg_away_xg]
        
        # Bounds to ensure positive xG values
        bounds = [(0.01, 5), (0.01, 5)]
        
        # Solve the optimization problem
        result = minimize(
            objective_function, 
            initial_guess, 
            method='L-BFGS-B', 
            bounds=bounds
        )
        
        # Return the optimized parameters (home_xg, away_xg)
        return result.x

    # def add_elo_xg_only(file_path, output_file=None, num_recent_games=6, home_advantage=35, league_home_xg=1.35, league_away_xg=1.1, def_elo=1500, def_f_factor=32):
    def add_elo_xg_only(dataframe, num_recent_games=6, home_advantage=35, league_home_xg=1.35, league_away_xg=1.1, def_elo=1500, def_f_factor=32):
        """
        Calculate expected goals based on Elo ratings and add only the xG values to the football dataset.
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file containing the football data
        output_file : str, optional
            Path to save the processed data (default: None, which appends '_xg_only' to original filename)
        num_recent_games : int
            Number of recent games to use for calculating league averages
        home_advantage : int
            Home advantage in Elo points
        league_home_xg : float
            Default league average for home team expected goals
        league_away_xg : float
            Default league average for away team expected goals
        def_elo : int
            Default Elo rating for teams
        def_f_factor : int
            K-factor for Elo rating updates
        
        Returns:
        --------
        pandas.DataFrame
            The enhanced dataframe with only the xG columns added
        """
        # Set default output file name if not provided
        # if output_file is None:
        #     output_file = file_path.replace('.xlsx', '_xg_only.xlsx')
        
        # Load and process the data
        # df = load_and_process_data(file_path)
        df = dataframe
        
        if df is None:
            context.log("Failed to load data. Exiting.")
            return None
        
        # Create a sequential index for chronological ordering
        df = df.reset_index(drop=True)
        
        # Initialize only the xG columns in the final dataframe
        output_columns = [
            'home_xg_elo',
            'away_xg_elo'
        ]
        
        # We'll still calculate the Elo ratings and probabilities temporarily
        temp_columns = [
            'home_team_elo',
            'away_team_elo',
            'home_win_probability',
            'draw_probability',
            'away_win_probability'
        ]
        
        # Initialize all temporary and output columns with default values
        all_columns = output_columns + temp_columns
        for col in all_columns:
            df[col] = 0.0
        
        # Analyze historical data to get league averages for xG
        # These will be used as initial values for the optimization
        league_avg_home_xg = league_home_xg  # Default if we can't calculate from data
        league_avg_away_xg = league_away_xg  # Default if we can't calculate from data
        
        # Try to calculate from data if we have enough completed matches
        completed_matches = df[df['status'].isin(['complete', 'finished'])]
        if len(completed_matches) >= num_recent_games:
            league_avg_home_xg = completed_matches['home_goals'].mean()
            league_avg_away_xg = completed_matches['away_goals'].mean()
            context.log(f"Calculated league averages - Home xG: {league_avg_home_xg:.2f}, Away xG: {league_avg_away_xg:.2f}")
        else:
            context.log(f"Using default league averages - Home xG: {league_avg_home_xg:.2f}, Away xG: {league_avg_away_xg:.2f}")
        
        # Process each season separately
        for season in df['season'].unique():
            season_df = df[df['season'] == season].copy()
            season_indices = season_df.index.tolist()
            
            # Dictionary to store team Elo ratings for this season
            elo_ratings = {}  # Dictionary to store team Elo ratings
            default_elo = def_elo  # Starting Elo rating for teams without history
            k_factor = def_f_factor  # K-factor determines how quickly ratings change
            
            # Home field advantage in Elo points
            home_advantage_elo = home_advantage
            
            # Initialize Elo ratings for teams in this season
            teams_in_season = set(season_df['home_team'].unique()) | set(season_df['away_team'].unique())
            for team in teams_in_season:
                elo_ratings[team] = default_elo
            
            # Process each match chronologically within the season
            for match_idx in season_indices:
                home_team = df.at[match_idx, 'home_team']
                away_team = df.at[match_idx, 'away_team']
                
                # Set current Elo ratings for this match (before updating)
                current_home_elo = elo_ratings.get(home_team, default_elo)
                current_away_elo = elo_ratings.get(away_team, default_elo)
                
                # Store Elo ratings temporarily (won't be included in final output)
                df.at[match_idx, 'home_team_elo'] = current_home_elo
                df.at[match_idx, 'away_team_elo'] = current_away_elo
                
                # Calculate win probabilities based on Elo with home advantage
                home_elo_adjusted = current_home_elo + home_advantage_elo
                elo_diff = (home_elo_adjusted - current_away_elo) / 400.0
                home_win_prob_raw = 1.0 / (1.0 + 10.0 ** (-elo_diff))
                
                # Calculate away win probability directly from Elo
                away_elo_adjusted = current_away_elo + home_advantage_elo  # If they were home
                away_elo_diff = (away_elo_adjusted - current_home_elo) / 400.0
                away_win_prob_raw = 1.0 / (1.0 + 10.0 ** (-away_elo_diff))
                
                # Calibrate draw probability 
                base_draw_prob = 0.28
                
                # Adjust draw probability based on how close the teams are
                elo_diff_abs = abs(current_home_elo - current_away_elo)
                draw_adjustment = max(0, 0.06 - (elo_diff_abs / 2000))
                draw_prob = base_draw_prob + draw_adjustment
                
                # Recalibrate win probabilities to account for draw and ensure all sum to 1
                remaining_prob = 1.0 - draw_prob
                
                # Calculate relative strengths of home and away teams
                total_win_prob_raw = home_win_prob_raw + away_win_prob_raw
                relative_home_strength = home_win_prob_raw / total_win_prob_raw
                relative_away_strength = away_win_prob_raw / total_win_prob_raw
                
                # Distribute the remaining probability according to relative strengths
                home_win_prob = relative_home_strength * remaining_prob
                away_win_prob = relative_away_strength * remaining_prob
                
                # Final check to ensure probabilities are in valid range and sum to 1
                total_prob = home_win_prob + draw_prob + away_win_prob
                if abs(total_prob - 1.0) > 0.0001:  # If not very close to 1
                    home_win_prob /= total_prob
                    draw_prob /= total_prob
                    away_win_prob /= total_prob
                
                # Store probabilities temporarily (won't be included in final output)
                df.at[match_idx, 'home_win_probability'] = home_win_prob
                df.at[match_idx, 'draw_probability'] = draw_prob
                df.at[match_idx, 'away_win_probability'] = away_win_prob
                
                # Calculate expected goals based on these probabilities
                try:
                    target_probs = (home_win_prob, draw_prob, away_win_prob)
                    home_xg, away_xg = solve_for_xg(target_probs, league_avg_home_xg, league_avg_away_xg)
                    
                    df.at[match_idx, 'home_xg_elo'] = round(home_xg, 2)
                    df.at[match_idx, 'away_xg_elo'] = round(away_xg, 2)
                except Exception as e:
                    context.log(f"Error calculating xG for match {match_idx}: {e}")
                    # Fallback calculation for xG based on relative team strength
                    home_strength_ratio = 10 ** (elo_diff)
                    
                    # More balanced xG values even when optimization fails
                    if home_strength_ratio > 1:  # Home team is stronger
                        df.at[match_idx, 'home_xg_elo'] = round(league_avg_home_xg * (home_strength_ratio ** 0.15), 2)
                        df.at[match_idx, 'away_xg_elo'] = round(league_avg_away_xg * (1 / home_strength_ratio ** 0.1), 2)
                    else:  # Away team is stronger
                        df.at[match_idx, 'home_xg_elo'] = round(league_avg_home_xg * (home_strength_ratio ** 0.1), 2)
                        df.at[match_idx, 'away_xg_elo'] = round(league_avg_away_xg * (1 / home_strength_ratio ** 0.15), 2)
                
                # Update Elo ratings after the match if it has been played
                if match_idx in df.index and df.at[match_idx, 'status'] in ['complete', 'finished']:
                    # Get match result
                    match_result = df.at[match_idx, 'full_time_result']
                    
                    # Only update if we have a result
                    if pd.notna(match_result):
                        # Convert result to score for Elo calculation
                        if match_result == 'H':
                            actual_score = 1.0  # Home win
                        elif match_result == 'A':
                            actual_score = 0.0  # Away win
                        else:  # Draw
                            actual_score = 0.5
                        
                        # Calculate expected score based on Elo difference (including draw probability)
                        expected_score = home_win_prob + (0.5 * draw_prob)
                        
                        # Calculate Elo updates
                        elo_change = k_factor * (actual_score - expected_score)
                        
                        # Update Elo ratings for next matches
                        elo_ratings[home_team] = current_home_elo + elo_change
                        elo_ratings[away_team] = current_away_elo - elo_change
        
        # Create a new DataFrame with only the original columns plus xG columns
        # This ensures we don't include the temporary Elo and probability columns
        original_columns = [col for col in df.columns if col not in all_columns]
        final_columns = original_columns + output_columns
        output_df = df[final_columns].copy()

        # output_df = output_df.drop(['full_time_result'], axis=1)
        
        # Save the enhanced dataframe with only the xG columns
        # output_df.to_excel(output_file, index=False)
        # context.log(f"Data with only xG values saved to {output_file}")
        
        return output_df

    # Example usage
    df_elo = add_elo_xg_only(clean_dataframe, num_recent_games=NUM_PREVIOUS_GAMES, 
                                    home_advantage=35, league_home_xg=1.35, league_away_xg=1.1,
                                    def_elo=1500, def_f_factor=32)

    # Save the enhanced dataframe with only the xG columns
    # df_elo.to_excel(OUTPUT_FILENAME, index=False)
    # context.log(f"Data with only xG values saved to {OUTPUT_FILENAME}")

    ###################################################################################################
    #########################     DIXON-COLES MODEL    ################################################

    def _get_team_matches_dixonCole(df, team):
        """Get all matches for a team (both home and away)"""
        return df[(df['home_team'] == team) | (df['away_team'] == team)]

    def _dixon_coles_correction(home_goals, away_goals, home_rate, away_rate, rho):
        """Dixon-Coles correction function for low scoring matches."""
        if home_goals == 0 and away_goals == 0:
            return 1 - (home_rate * away_rate * rho)
        elif home_goals == 1 and away_goals == 0:
            return 1 + (away_rate * rho)
        elif home_goals == 0 and away_goals == 1:
            return 1 + (home_rate * rho)
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1.0

    def _dixon_coles_loglikelihood(params, match_data, teams_attack, teams_defense, num_teams):
        """Calculate the negative log-likelihood for the Dixon-Coles model."""
        # Extract parameters
        home_advantage = params[0]
        rho = params[1]
        
        # Calculate negative log-likelihood for matches
        nll = 0.0
        
        for home_idx, away_idx, home_goals, away_goals in match_data:
            # Calculate expected goals
            home_rate = np.exp(teams_attack[home_idx] + teams_defense[away_idx] + home_advantage)
            away_rate = np.exp(teams_attack[away_idx] + teams_defense[home_idx])
            
            # Apply Dixon-Coles correction for low-scoring matches
            correction = _dixon_coles_correction(home_goals, away_goals, home_rate, away_rate, rho)
            
            # Poisson probability with Dixon-Coles correction
            home_prob = np.exp(-home_rate) * (home_rate ** home_goals) / np.math.factorial(home_goals)
            away_prob = np.exp(-away_rate) * (away_rate ** away_goals) / np.math.factorial(away_goals)
            
            # Add to negative log-likelihood
            if correction > 0:
                nll -= np.log(home_prob * away_prob * correction)
            else:
                # Avoid numerical issues
                nll += 10  # Penalize impossible scenarios
        
        # Add regularization to prevent extreme values
        regularization_strength = 0.1
        for team_idx in range(num_teams):
            nll += regularization_strength * (teams_attack[team_idx]**2 + teams_defense[team_idx]**2)
        
        return nll

    def _fit_dixon_coles_model(match_data, teams, initial_values=None, min_home_advantage=0.3):
        """
        Fit the Dixon-Coles model to match data.
        
        Parameters:
        -----------
        match_data : list of tuples
            List of (home_idx, away_idx, home_goals, away_goals) tuples
        teams : list
            List of team names
        initial_values : dict or None
            Dictionary of initial attack and defense values for teams
        min_home_advantage : float
            Minimum home advantage value to enforce (default: 0.3)
            
        Returns:
        --------
        tuple (attack_strengths, defense_strengths, home_advantage, rho)
            Fitted model parameters
        """
        num_teams = len(teams)
        teams_idx = {team: i for i, team in enumerate(teams)}
        
        # Initial values for team parameters
        if initial_values is None:
            # Default initial values
            teams_attack = {i: 0.0 for i in range(num_teams)}
            teams_defense = {i: 0.0 for i in range(num_teams)}
        else:
            # Use provided initial values
            teams_attack = {teams_idx[team]: values['attack'] for team, values in initial_values.items() if team in teams_idx}
            teams_defense = {teams_idx[team]: values['defense'] for team, values in initial_values.items() if team in teams_idx}
            
            # Set default values for any missing teams
            for i in range(num_teams):
                if i not in teams_attack:
                    teams_attack[i] = 0.0
                if i not in teams_defense:
                    teams_defense[i] = 0.0
        
        # Initial values for global parameters
        # Start with a higher initial home advantage to encourage the model to find values above our minimum
        initial_params = np.array([max(0.5, min_home_advantage), -0.1])  # home advantage and rho
        
        # Define objective function for the global parameters optimization
        def objective(params):
            return _dixon_coles_loglikelihood(params, match_data, teams_attack, teams_defense, num_teams)
        
        # Optimize global parameters with lower bound for home advantage
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=[(min_home_advantage, 1.0), (-1.0, 0.0)],  # home advantage [min,1], rho [-1,0]
            options={'maxiter': 100}
        )
        
        # Extract optimized parameters
        home_advantage = max(result.x[0], min_home_advantage)  # Enforce minimum even if optimizer went below
        rho = result.x[1]
        
        # Create a Poisson regression dataset for estimating team parameters
        X_data = []
        y_data = []
        
        for home_idx, away_idx, home_goals, away_goals in match_data:
            # For home goals
            x_row = np.zeros(2 * num_teams + 1)
            x_row[home_idx] = 1  # Home attack
            x_row[num_teams + away_idx] = 1  # Away defense
            x_row[-1] = 1  # Home advantage
            X_data.append(x_row)
            y_data.append(home_goals)
            
            # For away goals
            x_row = np.zeros(2 * num_teams + 1)
            x_row[away_idx] = 1  # Away attack
            x_row[num_teams + home_idx] = 1  # Home defense
            X_data.append(x_row)
            y_data.append(away_goals)
        
        # Fit Poisson regression model
        X = np.array(X_data)
        y = np.array(y_data)
        
        try:
            model = sm.GLM(y, X, family=sm.families.Poisson())
            result = model.fit(disp=0)
            
            # Extract parameters
            coeffs = result.params
            
            # Update team parameters
            for i in range(num_teams):
                teams_attack[i] = coeffs[i]
                teams_defense[i] = coeffs[num_teams + i]
            
            # Home advantage from Poisson model, but ensure it's above minimum
            home_advantage = max(coeffs[-1], min_home_advantage)
        except:
            context.log("  Error fitting Poisson model, using partial estimates")
        
        # Ensure identifiability by centering the attack and defense parameters
        avg_attack = sum(teams_attack.values()) / num_teams
        avg_defense = sum(teams_defense.values()) / num_teams
        
        for i in range(num_teams):
            teams_attack[i] -= avg_attack
            teams_defense[i] -= avg_defense
        
        # Convert parameters to team ratings
        attack_strengths = {teams[i]: teams_attack[i] for i in range(num_teams)}
        defense_strengths = {teams[i]: teams_defense[i] for i in range(num_teams)}
        
        return attack_strengths, defense_strengths, home_advantage, rho

    # Function to calculate expected goals using team ratings
    def calculate_expected_goals_DixonCole(home_team, away_team, attack_params, defense_params, home_advantage=1.3):
        """
        Calculate expected goals for a match using team attack and defense parameters.
        
        Parameters:
        -----------
        home_team, away_team : str
            Team names
        attack_params, defense_params : dict
            Dictionaries containing attack and defense parameters for all teams
        home_advantage : float
            Home advantage factor (default: 1.2)
        
        Returns:
        --------
        tuple
            (expected_home_goals, expected_away_goals)
        """
        # Get parameters - default to middle value (1.0) if not available
        home_attack = attack_params.get(home_team, 1.0)
        home_defense = defense_params.get(home_team, 1.0)
        away_attack = attack_params.get(away_team, 1.0)
        away_defense = defense_params.get(away_team, 1.0)
        
        # For defense_factor, higher values mean better defense (fewer goals conceded)
        # We need to convert defense rating to a factor that reduces expected goals
        home_defense_factor = (3.0 - home_defense) / 3.0
        away_defense_factor = (3.0 - away_defense) / 3.0
        
        # Calculate expected goals
        # Home team expected goals = home team attack * away team defense factor * home advantage
        # Away team expected goals = away team attack * home team defense factor
        expected_home_goals = home_attack * away_defense_factor * home_advantage
        expected_away_goals = away_attack * home_defense_factor
        
        # Ensure expected goals are within a reasonable range (0-3)
        expected_home_goals = min(max(expected_home_goals, 0.0), 3.0)
        expected_away_goals = min(max(expected_away_goals, 0.0), 3.0)
        
        return expected_home_goals, expected_away_goals

    def add_dixon_coles_ratings(dataframe, num_recent_games=6, min_home_advantage=0.3, home_advantage_factor=1.3):
        """
        Add Dixon-Coles model rating columns to the football dataset.
        Ratings are explicitly scaled to be positive and within the range 0-5.
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file containing the football data
        num_recent_games : int, optional
            Number of recent games to consider for calculations (default: 6)
        min_home_advantage : float, optional
            Minimum home advantage value to enforce (default: 0.3)
        home_advantage_factor : float, optional
            Home advantage factor for expected goals calculation (default: 1.2)
        output_file : str, optional
            Path to save the processed data (default: None, which appends '_dc' to original filename)
        
        Returns:
        --------
        pandas.DataFrame
            The enhanced dataframe with Dixon-Coles rating columns
        """
        # Set default output file name if not provided
        # if output_file is None:
        #     output_file = file_path.replace('.xlsx', '_dc_fixed.xlsx')
        
        # Load and process the data
        # df = load_and_process_data(file_path)
        df = dataframe

        if df is None:
            context.log("Failed to load data. Exiting.")
            return None
        
        # Create a sequential index for chronological ordering
        df = df.reset_index(drop=True)
        
        # Initialize Dixon-Coles rating columns
        dc_columns = [
            'home_team_attack',
            'home_team_defense',
            'away_team_attack',
            'away_team_defense',
            'dc_home_advantage',
            'home_xg_dc',  # New column for home team expected goals
            'away_xg_dc'   # New column for away team expected goals
        ]
        
        # Initialize all columns with default values
        for col in dc_columns:
            if col.endswith('_attack') or col.endswith('_defense'):
                df[col] = 0.5  # Default rating of 0.5
            elif col == 'dc_home_advantage':
                df[col] = min_home_advantage  # Set to minimum home advantage
            elif col.endswith('_xg_dc'):
                df[col] = 0.0  # Default expected goals of 0.0
            else:
                df[col] = 0.0
        
        # Process each season separately
        seasons = sorted(df['season'].unique())
        for i, season in enumerate(seasons):
            context.log(f"Processing season: {season}")
            season_df = df[df['season'] == season].copy()
            season_indices = season_df.index.tolist()
            
            # Dixon-Coles model parameters for this season
            team_attack = {}  # Dictionary to store attack parameters
            team_defense = {}  # Dictionary to store defense parameters
            home_advantage = min_home_advantage  # Initialize with minimum home advantage
            rho = 0.0  # Dixon-Coles correlation parameter (still used internally but not exported)
            
            # Initialize ratings for teams in this season
            teams_in_season = list(set(season_df['home_team'].unique()) | set(season_df['away_team'].unique()))
            for team in teams_in_season:
                team_attack[team] = 0.0
                team_defense[team] = 0.0
            
            # Track the model fitting frequency
            last_fit_index = -1
            
            # Adjust fit interval based on season length
            season_length = len(season_indices)
            fit_interval = max(num_recent_games, int(season_length / max(1, season_length/num_recent_games)))
            
            # Adjust minimum matches needed based on season position
            if i == 0:
                # First season - start fitting earlier
                min_matches_needed = num_recent_games
            else:
                # Subsequent seasons - standard threshold
                min_matches_needed = num_recent_games
            
            # Process each match chronologically within the season
            for match_idx_position, match_idx in enumerate(season_indices):
                home_team = df.at[match_idx, 'home_team']
                away_team = df.at[match_idx, 'away_team']
                
                # Get all previous matches in this season up to this match
                prev_season_matches = season_df[season_df.index < match_idx]
                
                # Check if we should recalculate ratings
                enough_matches = len(prev_season_matches) >= min_matches_needed
                enough_new_matches = match_idx_position - last_fit_index >= fit_interval
                
                if enough_matches and (enough_new_matches or match_idx_position == len(season_indices) - 1):
                    context.log(f"  Fitting model at match {match_idx_position+1}/{len(season_indices)}")
                    last_fit_index = match_idx_position
                    
                    # Create a list of match data for fitting
                    match_data = []
                    
                    # Teams that appear in recent matches
                    teams_with_recent_matches = set()
                    
                    # For each team, get their most recent matches
                    for team in teams_in_season:
                        team_matches = _get_team_matches_dixonCole(prev_season_matches, team)
                        
                        # Only use the most recent num_recent_games for each team
                        if len(team_matches) > 0:
                            recent_team_matches = team_matches.tail(min(len(team_matches), num_recent_games))
                            
                            # Add to the set of teams with recent matches
                            teams_with_recent_matches.add(team)
                            
                            # Add each match to the data if not already added
                            for _, match in recent_team_matches.iterrows():
                                home_team_match = match['home_team']
                                away_team_match = match['away_team']
                                
                                # Add both teams to the set
                                teams_with_recent_matches.add(home_team_match)
                                teams_with_recent_matches.add(away_team_match)
                    
                    # Filter to only teams with recent matches
                    teams_to_fit = list(teams_with_recent_matches)
                    teams_idx = {team: i for i, team in enumerate(teams_to_fit)}
                    
                    # Prepare match data for model fitting
                    for _, match in prev_season_matches.iterrows():
                        home_team_match = match['home_team']
                        away_team_match = match['away_team']
                        
                        # Only include matches between teams in our index
                        if home_team_match in teams_idx and away_team_match in teams_idx:
                            home_idx = teams_idx[home_team_match]
                            away_idx = teams_idx[away_team_match]
                            home_goals = match['home_goals']
                            away_goals = match['away_goals']
                            
                            match_data.append((home_idx, away_idx, home_goals, away_goals))
                    
                    if len(match_data) >= (num_recent_games-1) and len(teams_to_fit) >= (num_recent_games-1):
                        # Prepare initial values
                        initial_values = {}
                        for team in teams_to_fit:
                            if team in team_attack and team in team_defense:
                                initial_values[team] = {
                                    'attack': team_attack[team],
                                    'defense': team_defense[team]
                                }
                        
                        # Fit the Dixon-Coles model with minimum home advantage
                        try:
                            attack, defense, home_advantage, rho = _fit_dixon_coles_model(
                                match_data, teams_to_fit, initial_values, min_home_advantage
                            )
                            
                            # Update team parameters
                            for team, att in attack.items():
                                team_attack[team] = att
                            
                            for team, defs in defense.items():
                                team_defense[team] = defs
                            
                            context.log(f"  Model fit successful. Home advantage: {home_advantage:.2f}")
                        except Exception as e:
                            context.log(f"  Error fitting Dixon-Coles model: {e}")
                            # If optimization fails, keep existing ratings but ensure home advantage is at least minimum
                            home_advantage = max(home_advantage, min_home_advantage)
                
                # Set current Dixon-Coles ratings for this match
                current_home_attack = team_attack.get(home_team, 0.0)
                current_home_defense = team_defense.get(home_team, 0.0)
                current_away_attack = team_attack.get(away_team, 0.0)
                current_away_defense = team_defense.get(away_team, 0.0)
                
                # Scale ratings to 0-3 range
                all_attack_values = list(team_attack.values())
                all_defense_values = list(team_defense.values())
                
                if len(all_attack_values) >= 2:
                    # Find min and max values
                    min_attack = min(all_attack_values)
                    max_attack = max(all_attack_values)
                    min_defense = min(all_defense_values)
                    max_defense = max(all_defense_values)
                    
                    # Range for scaling
                    attack_range = max_attack - min_attack if max_attack > min_attack else 1.0
                    defense_range = max_defense - min_defense if max_defense > min_defense else 1.0
                    
                    # Scale to 0-3 range and shift to ensure minimum value is 0.5
                    # For attack, higher raw value = better attack
                    home_attack_scaled = 0.5 + 2.5 * (current_home_attack - min_attack) / attack_range
                    away_attack_scaled = 0.5 + 2.5 * (current_away_attack - min_attack) / attack_range
                    
                    # For defense, lower raw value = better defense, so invert
                    home_defense_scaled = 0.5 + 2.5 * (max_defense - current_home_defense) / defense_range
                    away_defense_scaled = 0.5 + 2.5 * (max_defense - current_away_defense) / defense_range
                    
                    # Clip to ensure 0.5-3 range even if distribution is skewed
                    home_attack_scaled = min(max(home_attack_scaled, 0.5), 3.0)
                    home_defense_scaled = min(max(home_defense_scaled, 0.5), 3.0)
                    away_attack_scaled = min(max(away_attack_scaled, 0.5), 3.0)
                    away_defense_scaled = min(max(away_defense_scaled, 0.5), 3.0)
                    
                    # Create a dictionary of scaled attack and defense values for expected goals calculation
                    attack_params = {
                        home_team: home_attack_scaled,
                        away_team: away_attack_scaled
                    }
                    
                    defense_params = {
                        home_team: home_defense_scaled,
                        away_team: away_defense_scaled
                    }
                    
                    # Calculate expected goals using the formula from the first script
                    home_expected_goals, away_expected_goals = calculate_expected_goals_DixonCole(
                        home_team, 
                        away_team, 
                        attack_params, 
                        defense_params, 
                        home_advantage_factor
                    )
                else:
                    # Not enough teams for scaling, use default value
                    home_attack_scaled = 0.5
                    home_defense_scaled = 0.5
                    away_attack_scaled = 0.5
                    away_defense_scaled = 0.5
                    home_expected_goals = 0.0
                    away_expected_goals = 0.0
                
                # Store ratings in the dataframe
                df.at[match_idx, 'home_team_attack'] = home_attack_scaled
                df.at[match_idx, 'home_team_defense'] = home_defense_scaled
                df.at[match_idx, 'away_team_attack'] = away_attack_scaled
                df.at[match_idx, 'away_team_defense'] = away_defense_scaled
                df.at[match_idx, 'dc_home_advantage'] = home_advantage
                df.at[match_idx, 'home_xg_dc'] = round(home_expected_goals, 2)
                df.at[match_idx, 'away_xg_dc'] = round(away_expected_goals, 2)
        
        # Round all rating columns to 2 decimal places
        for col in dc_columns:
            df[col] = df[col].round(2)
        
        # Final check to ensure all home advantage values are at least the minimum
        df['dc_home_advantage'] = np.maximum(df['dc_home_advantage'], min_home_advantage)

        # drop unwanted columns
        df = df.drop(['home_team_attack', 'home_team_defense', 'away_team_attack', 'away_team_defense', 'dc_home_advantage'], axis=1)
        
        # Save the enhanced dataframe
        # df.to_excel(output_file, index=False)
        # context.log(f"Data with Dixon-Coles ratings saved to {output_file}")
        
        return df

    # Example usage
    df_dc = add_dixon_coles_ratings(
        df_elo,
        num_recent_games=NUM_PREVIOUS_GAMES,
        min_home_advantage=MIN_HOME_ADVANTAGE,  # Setting minimum home advantage
        home_advantage_factor=HOME_ADVANTAGE,  # Home advantage factor for expected goals calculation
    )


    # Save the enhanced dataframe with only the xG columns
    # df_dc.to_excel(OUTPUT_FILENAME, index=False)
    # context.log(f"Data with only xG values saved to {OUTPUT_FILENAME}")


    ############################################################################################################
    #################################     Bradley-Terry MODEL    ################################################


    def _get_team_matches_bradleyTerry(df, team):
        """Get all matches for a team (both home and away)"""
        return df[(df['home_team'] == team) | (df['away_team'] == team)]

    def _bradley_terry_loglikelihood(params, match_data, teams_idx, teams_count):
        """
        Calculate the negative log-likelihood for the Bradley-Terry model with regularization.
        
        Parameters:
        -----------
        params : array
            Array of parameters: team strengths followed by home advantage
        match_data : list of tuples
            List of (home_idx, away_idx, home_win, draw) tuples
        teams_idx : dict
            Mapping of team names to index in strengths array
        teams_count : int
            Number of teams
            
        Returns:
        --------
        float
            Negative log-likelihood with regularization
        """
        # Separate team strengths and home advantage
        strengths = params[:teams_count]
        home_advantage = params[-1]
        
        # Add constraint to ensure sum of strengths is 0 (identifiability constraint)
        strengths = strengths - np.mean(strengths)
        
        # Calculate negative log-likelihood for matches
        nll = 0.0
        for home_idx, away_idx, home_win, draw in match_data:
            # Get team strengths
            s_i = strengths[home_idx] + home_advantage
            s_j = strengths[away_idx]
            
            # Difference in strengths determines win probability
            diff = s_i - s_j
            
            # Modified logistic function for probability calculation
            p_i = 1.0 / (1.0 + np.exp(-diff))
            
            # For draws, use an ordered logit model
            # Higher probability of draw when teams are more evenly matched
            p_draw = max(0.0, 1.0 - abs(p_i - 0.5) * 2.0)  # Simple draw model
            p_draw = min(p_draw, 0.5)  # Cap the draw probability
            
            # Adjust win/loss probabilities
            p_i_win = p_i * (1.0 - p_draw)
            p_j_win = (1.0 - p_i) * (1.0 - p_draw)
            
            # Add to negative log-likelihood based on actual result
            if draw:
                nll -= np.log(max(p_draw, 1e-10))
            elif home_win:
                nll -= np.log(max(p_i_win, 1e-10))
            else:
                nll -= np.log(max(p_j_win, 1e-10))
        
        # Add regularization to prevent extreme values (L2 regularization)
        regularization_strength = 0.1
        nll += regularization_strength * np.sum(strengths**2)
        
        return nll

    def add_bradley_terry_ratings(dataframe, num_recent_games=6):
        """
        Add Bradley-Terry model rating columns to the football dataset.
        Ratings are scaled to be positive and within the range 0-5 with average at 1.0.
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file containing the football data
        num_recent_games : int, optional
            Number of recent games to consider for calculations (default: 6)
        output_file : str, optional
            Path to save the processed data (default: None, which appends '_bt' to original filename)
        
        Returns:
        --------
        pandas.DataFrame
            The enhanced dataframe with Bradley-Terry rating columns
        """
        # Set default output file name if not provided
        # if output_file is None:
        #     output_file = file_path.replace('.xlsx', '_bt_fixed.xlsx')
        
        # Load and process the data
        # df = load_and_process_data(file_path)
        df = dataframe

        if df is None:
            context.log("Failed to load data. Exiting.")
            return None
        
        # Create a sequential index for chronological ordering
        df = df.reset_index(drop=True)
        
        # Initialize Bradley-Terry rating columns
        bt_columns = [
            'home_xg_bt',
            'away_xg_bt',
            'bt_home_advantage'
        ]
        
        # Initialize all columns with default values
        for col in bt_columns:
            df[col] = 0.0
        
        # Process each season separately
        for season in df['season'].unique():
            context.log(f"Processing season: {season}")
            season_df = df[df['season'] == season].copy()
            season_indices = season_df.index.tolist()
            
            # Bradley-Terry model parameters
            bt_ratings = {}  # Dictionary to store Bradley-Terry ratings
            bt_home_advantage = 0.0  # Home advantage parameter for Bradley-Terry model
            
            # Initialize ratings for teams in this season
            teams_in_season = list(set(season_df['home_team'].unique()) | set(season_df['away_team'].unique()))
            for team in teams_in_season:
                bt_ratings[team] = 0.0  # Start with neutral rating
            
            # Create a mapping from team names to indices
            teams_idx = {team: i for i, team in enumerate(teams_in_season)}
            
            # Track the model fitting frequency - don't need to fit every match
            last_fit_index = -1
            # fit_interval = max(5, len(season_df) // 20)  # Fit approximately 20 times per season
            fit_interval = max(num_recent_games, len(season_df) // (len(season_df)/num_recent_games))  # Fit approximately (len(season_df)/num_recent_games) times per season
            
            # Process each match chronologically within the season
            for match_idx_position, match_idx in enumerate(season_indices):
                home_team = df.at[match_idx, 'home_team']
                away_team = df.at[match_idx, 'away_team']
                
                # Get all previous matches in this season up to this match
                prev_season_matches = season_df[season_df.index < match_idx]
                
                # Recalculate Bradley-Terry ratings periodically
                # enough_matches = len(prev_season_matches) >= 15  # Need matches for reliable Bradley-Terry
                enough_matches = len(prev_season_matches) >= num_recent_games  # Need matches for reliable Bradley-Terry: num_recent_games
                enough_new_matches = match_idx_position - last_fit_index >= fit_interval
                
                if enough_matches and (enough_new_matches or match_idx_position == len(season_indices) - 1):
                    context.log(f"  Fitting model at match {match_idx_position+1}/{len(season_indices)}")
                    last_fit_index = match_idx_position
                    
                    # Create a list of match data for optimization
                    match_data = []
                    
                    # Create a set to track teams with recent matches
                    teams_with_recent_matches = set()
                    
                    # For each team, get their most recent matches
                    for team in teams_in_season:
                        team_matches = _get_team_matches_bradleyTerry(prev_season_matches, team)
                        
                        # Only use the most recent num_recent_games for each team
                        recent_team_matches = team_matches.tail(min(len(team_matches), num_recent_games))
                        
                        # Add to the set of teams with recent matches
                        if len(recent_team_matches) > 0:
                            teams_with_recent_matches.add(team)
                            
                            # Add each match to match_data if not already added
                            for _, match in recent_team_matches.iterrows():
                                home_team_match = match['home_team']
                                away_team_match = match['away_team']
                                
                                # Both teams need to be in our index
                                if home_team_match in teams_idx and away_team_match in teams_idx:
                                    home_idx = teams_idx[home_team_match]
                                    away_idx = teams_idx[away_team_match]
                                    home_win = match['full_time_result'] == 'H'
                                    draw = match['full_time_result'] == 'D'
                                    
                                    # Check if this match is already in match_data
                                    match_tuple = (home_idx, away_idx, home_win, draw)
                                    if match_tuple not in match_data:
                                        match_data.append(match_tuple)
                    
                    # if len(match_data) >= 10:  # Ensure enough matches for optimization
                    if len(match_data) >= num_recent_games:  # Ensure enough matches for optimization: num_recent_games
                        # Initial values - start with current ratings if available
                        initial_strengths = np.zeros(len(teams_in_season))
                        for team, idx in teams_idx.items():
                            initial_strengths[idx] = bt_ratings.get(team, 0.0)
                        
                        # Add home advantage parameter
                        initial_params = np.append(initial_strengths, bt_home_advantage if bt_home_advantage != 0 else 0.1)
                        
                        # Optimize with constraints and regularization
                        try:
                            result = minimize(
                                lambda params: _bradley_terry_loglikelihood(
                                    params, match_data, teams_idx, len(teams_in_season)
                                ),
                                initial_params,
                                method='L-BFGS-B',  # More stable method
                                bounds=[(-3, 3)] * len(teams_in_season) + [(0, 1)],  # Bounds to prevent extreme values
                                options={'maxiter': 1000}
                            )
                            
                            # Extract results
                            optimized_strengths = result.x[:len(teams_in_season)]
                            bt_home_advantage = result.x[-1]
                            
                            # Center strengths to ensure identifiability
                            optimized_strengths = optimized_strengths - np.mean(optimized_strengths)
                            
                            # Transform ratings to 0-5 scale with average at 1.0
                            # First, normalize to make variance approximately 0.5
                            if len(optimized_strengths) > 1 and np.std(optimized_strengths) > 0:
                                optimized_strengths = optimized_strengths / (2 * np.std(optimized_strengths))
                            
                            # Then shift and scale to 0-5 range with average at 1.0
                            optimized_strengths = 1.0 + optimized_strengths  # Center at 1.0
                            
                            # Clip to ensure range bounds (0-5)
                            optimized_strengths = np.clip(optimized_strengths, 0, 5)
                            
                            # Update team ratings
                            for team, idx in teams_idx.items():
                                bt_ratings[team] = optimized_strengths[idx]
                            
                            context.log(f"  Model fit successful. Home advantage: {bt_home_advantage:.2f}")
                        except Exception as e:
                            context.log(f"  Error fitting Bradley-Terry model: {e}")
                            # If optimization fails, keep existing ratings
                
                # Set current Bradley-Terry ratings for this match
                current_home_bt = bt_ratings.get(home_team, 1.0)  # Default to average (1.0) if not available
                current_away_bt = bt_ratings.get(away_team, 1.0)
                
                # Store ratings in the dataframe
                df.at[match_idx, 'home_xg_bt'] = current_home_bt
                df.at[match_idx, 'away_xg_bt'] = current_away_bt
                df.at[match_idx, 'bt_home_advantage'] = bt_home_advantage
        
        # Round all rating columns to 2 decimal places
        for col in bt_columns:
            df[col] = df[col].round(2)
        
        # drop unwanted columns
        df = df.drop(['bt_home_advantage'], axis=1)

        # Save the enhanced dataframe
        # df.to_excel(output_file, index=False)
        # context.log(f"Data with Bradley-Terry ratings saved to {output_file}")
        
        return df

    # Example usage
    df_bt = add_bradley_terry_ratings(df_dc, num_recent_games=NUM_PREVIOUS_GAMES)



    # Save the enhanced dataframe with only the xG columns
    # df_bt.to_excel(OUTPUT_FILENAME, index=False)
    # context.log(f"Data with only xG values saved to {OUTPUT_FILENAME}")


    #################################################################################################################
    #################################     PYTHAGOREAN EXPECTATION    ################################################

    def _calculate_pythagorean_expectation(goals_scored, goals_conceded, exponent=1.83):
        """
        Calculate the Pythagorean expectation (expected win percentage) for a team.
        
        Parameters:
        -----------
        goals_scored : float
            Number of goals scored by the team
        goals_conceded : float
            Number of goals conceded by the team
        exponent : float
            Pythagorean exponent (default: 1.83, which is commonly used for soccer)
        
        Returns:
        --------
        float
            Expected win percentage (0-1)
        """
        # Avoid division by zero
        if goals_scored == 0 and goals_conceded == 0:
            return 0.5  # Neutral expectation
        elif goals_conceded == 0:
            return 1.0  # Perfect expectation
        
        # Calculate Pythagorean expectation
        expectation = goals_scored ** exponent / (goals_scored ** exponent + goals_conceded ** exponent)
        return expectation

    def _scale_to_range(value, old_min, old_max, new_min, new_max):
        """
        Scale a value from one range to another.
        
        Parameters:
        -----------
        value : float
            Value to scale
        old_min, old_max : float
            Original range
        new_min, new_max : float
            Target range
        
        Returns:
        --------
        float
            Scaled value
        """
        # Check if old range is valid
        if old_max == old_min:
            return (new_max + new_min) / 2  # Return midpoint of new range
        
        # Perform scaling
        scaled_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        
        # Ensure value is within new range
        return min(max(scaled_value, new_min), new_max)

    def add_pythagorean_ratings(dataframe, num_recent_games=6, exponent=1.83):
        """
        Add Pythagorean model ratings to the football dataset.
        All ratings are scaled to the 0.5-3 range.
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file containing the football data
        num_recent_games : int, optional
            Number of recent games to consider for calculations (default: 6)
        exponent : float, optional
            Pythagorean exponent (default: 1.83, common for soccer)
        output_file : str, optional
            Path to save the processed data (default: None, which appends '_pyth' to original filename)
        
        Returns:
        --------
        pandas.DataFrame
            The enhanced dataframe with Pythagorean rating columns
        """
        # Set default output file name if not provided
        # if output_file is None:
        #     output_file = file_path.replace('.xlsx', '_pyth_0.5_3.xlsx')
        
        # Load and process the data
        # df = load_and_process_data(file_path)
        df = dataframe
        
        if df is None:
            context.log("Failed to load data. Exiting.")
            return None
        
        # Create a sequential index for chronological ordering
        df = df.reset_index(drop=True)
        
        # Initialize Pythagorean model columns
        pyth_columns = [
            'home_xg_pyth',
            'away_xg_pyth',
            'home_team_luck_factor'
        ]
        
        # Initialize all columns with default values in 0.5-3 range
        for col in pyth_columns:
            if col == 'home_team_luck_factor':
                df[col] = 1.5  # Neutral luck factor (middle of 0.5-3 range)
            else:
                df[col] = 1.5  # Neutral team strength (middle of 0.5-3 range)
        
        # Process each season separately
        seasons = sorted(df['season'].unique())
        for i, season in enumerate(seasons):
            context.log(f"\nProcessing season: {season}")
            season_df = df[df['season'] == season].copy()
            season_indices = season_df.index.tolist()
            
            # Track match counts to help diagnose the issue
            season_match_count = 0
            teams_with_ratings = set()
            team_match_counts = {}
            
            # Initialize team data for better early-season estimates
            teams_in_season = list(set(season_df['home_team'].unique()) | set(season_df['away_team'].unique()))
            for team in teams_in_season:
                team_match_counts[team] = 0
            
            # Lower minimum match threshold to ensure more teams get ratings earlier
            # min_matches_needed = 2  # Lowered from 3/5 to get more teams rated early
            min_matches_needed = num_recent_games  # Lowered from 3/5 to get more teams rated early: num_recent_games
            
            # Process each match chronologically within the season
            for match_idx_position, match_idx in enumerate(season_indices):
                home_team = df.at[match_idx, 'home_team']
                away_team = df.at[match_idx, 'away_team']
                season_match_count += 1
                
                # Get all previous matches in this season up to this match
                prev_season_matches = season_df[season_df.index < match_idx]
                
                # Get team-specific match histories
                home_team_matches = prev_season_matches[(prev_season_matches['home_team'] == home_team) | 
                                                        (prev_season_matches['away_team'] == home_team)]
                away_team_matches = prev_season_matches[(prev_season_matches['home_team'] == away_team) | 
                                                        (prev_season_matches['away_team'] == away_team)]
                
                # Update match counts
                team_match_counts[home_team] = len(home_team_matches)
                team_match_counts[away_team] = len(away_team_matches)
                
                # Calculate home team statistics directly from the match history
                if len(home_team_matches) > 0:
                    # Initialize counters
                    home_team_goals_scored = 0
                    home_team_goals_conceded = 0
                    home_team_wins = 0
                    home_team_draws = 0
                    
                    # If very early in the season, we'll use all matches
                    # Otherwise, use only the most recent matches
                    if len(home_team_matches) <= min_matches_needed:
                        # Use all available matches
                        recent_home_matches = home_team_matches
                    else:
                        # Get only the most recent matches
                        recent_home_matches = home_team_matches.tail(min(len(home_team_matches), num_recent_games))
                    
                    # Process each match
                    for _, match in recent_home_matches.iterrows():
                        # Check if team played at home or away
                        if match['home_team'] == home_team:
                            # Team played at home
                            home_team_goals_scored += match['home_goals']
                            home_team_goals_conceded += match['away_goals']
                            
                            if match['full_time_result'] == 'H':
                                home_team_wins += 1
                            elif match['full_time_result'] == 'D':
                                home_team_draws += 1
                        else:
                            # Team played away
                            home_team_goals_scored += match['away_goals']
                            home_team_goals_conceded += match['home_goals']
                            
                            if match['full_time_result'] == 'A':
                                home_team_wins += 1
                            elif match['full_time_result'] == 'D':
                                home_team_draws += 1
                    
                    # Calculate actual win percentage (including draws as half wins)
                    actual_win_pct = (home_team_wins + home_team_draws * 0.5) / len(recent_home_matches)
                    
                    # Calculate Pythagorean expectation
                    expected_win_pct = _calculate_pythagorean_expectation(
                        home_team_goals_scored, home_team_goals_conceded, exponent
                    )
                    
                    # Calculate luck factor
                    home_team_luck = actual_win_pct - expected_win_pct
                    
                    # Scale expected win percentage to 0.5-3 range for team strength
                    # The mapping is: win% 0.0 -> 0.5, win% 0.5 -> 1.75, win% 1.0 -> 3.0
                    home_strength = _scale_to_range(expected_win_pct, 0.0, 1.0, 0.5, 3.0)
                    
                    # Scale luck factor to 0.5-3 range
                    # A neutral luck factor (no luck) should be 1.75 (middle of 0.5-3)
                    # Negative luck (actual < expected) -> 0.5-1.75
                    # Positive luck (actual > expected) -> 1.75-3.0
                    # Luck range is typically -0.3 to 0.3, so scale accordingly
                    home_luck_factor = _scale_to_range(home_team_luck, -0.3, 0.3, 0.5, 3.0)
                    
                    # context.log debug info for the first match for each team or periodically
                    if home_team not in teams_with_ratings or match_idx_position % 100 == 0:
                        context.log(f"  Match {season_match_count}: {home_team} vs {away_team}")
                        context.log(f"    {home_team} matches so far: {len(home_team_matches)}")
                        context.log(f"    Used for calculation: {len(recent_home_matches)} matches")
                        context.log(f"    Goals: Scored={home_team_goals_scored}, Conceded={home_team_goals_conceded}")
                        context.log(f"    Expected Win%={expected_win_pct:.3f}, Actual Win%={actual_win_pct:.3f}")
                        context.log(f"    Strength={home_strength:.2f}, Luck={home_luck_factor:.2f}")
                    
                    # Update values even if below the old threshold, as long as team has played at least 1 match
                    df.at[match_idx, 'home_xg_pyth'] = round(home_strength, 2)
                    df.at[match_idx, 'home_team_luck_factor'] = round(home_luck_factor, 2)
                    
                    # Track teams that have received ratings
                    teams_with_ratings.add(home_team)
                else:
                    # No previous matches, keep default values or use league averages
                    if match_idx_position % 100 == 0:
                        context.log(f"  Match {season_match_count}: {home_team} has no previous matches")
                
                # Same process for away team
                if len(away_team_matches) > 0:
                    # Initialize counters
                    away_team_goals_scored = 0
                    away_team_goals_conceded = 0
                    away_team_wins = 0
                    away_team_draws = 0
                    
                    # If very early in the season, we'll use all matches
                    # Otherwise, use only the most recent matches
                    if len(away_team_matches) <= min_matches_needed:
                        # Use all available matches
                        recent_away_matches = away_team_matches
                    else:
                        # Get only the most recent matches
                        recent_away_matches = away_team_matches.tail(min(len(away_team_matches), num_recent_games))
                    
                    # Process each match
                    for _, match in recent_away_matches.iterrows():
                        # Check if team played at home or away
                        if match['home_team'] == away_team:
                            # Team played at home
                            away_team_goals_scored += match['home_goals']
                            away_team_goals_conceded += match['away_goals']
                            
                            if match['full_time_result'] == 'H':
                                away_team_wins += 1
                            elif match['full_time_result'] == 'D':
                                away_team_draws += 1
                        else:
                            # Team played away
                            away_team_goals_scored += match['away_goals']
                            away_team_goals_conceded += match['home_goals']
                            
                            if match['full_time_result'] == 'A':
                                away_team_wins += 1
                            elif match['full_time_result'] == 'D':
                                away_team_draws += 1
                    
                    # Calculate actual win percentage (including draws as half wins)
                    actual_win_pct = (away_team_wins + away_team_draws * 0.5) / len(recent_away_matches)
                    
                    # Calculate Pythagorean expectation
                    expected_win_pct = _calculate_pythagorean_expectation(
                        away_team_goals_scored, away_team_goals_conceded, exponent
                    )
                    
                    # Scale expected win percentage to 0.5-3 range for team strength
                    away_strength = _scale_to_range(expected_win_pct, 0.0, 1.0, 0.5, 3.0)
                    
                    # context.log debug info for the first match for each team
                    if away_team not in teams_with_ratings and (match_idx_position < 10 or match_idx_position % 100 == 0):
                        context.log(f"    {away_team} matches so far: {len(away_team_matches)}")
                        context.log(f"    Used for calculation: {len(recent_away_matches)} matches")
                        context.log(f"    Goals: Scored={away_team_goals_scored}, Conceded={away_team_goals_conceded}")
                        context.log(f"    Expected Win%={expected_win_pct:.3f}, Actual Win%={actual_win_pct:.3f}")
                        context.log(f"    Strength={away_strength:.2f}")
                    
                    # Update away team strength even if below old threshold
                    df.at[match_idx, 'away_xg_pyth'] = round(away_strength, 2)
                    
                    # Track teams that have received ratings
                    teams_with_ratings.add(away_team)
                else:
                    # No previous matches
                    if match_idx_position % 100 == 0 and match_idx_position < 20:
                        context.log(f"    {away_team} has no previous matches")
            
            # context.log summary statistics for this season
            context.log(f"\nSeason {season} summary:")
            context.log(f"  Total matches in season: {season_match_count}")
            context.log(f"  Teams with ratings: {len(teams_with_ratings)} of {len(teams_in_season)}")
            
            # context.log match counts for each team to diagnose slow rating assignments
            context.log("\nTeam match counts:")
            sorted_teams = sorted(team_match_counts.items(), key=lambda x: x[1], reverse=True)
            for team, count in sorted_teams:
                context.log(f"  {team}: {count} matches")
        
        # Final check to ensure all values are within the 0.5-3 range
        for col in pyth_columns:
            df[col] = df[col].apply(lambda x: min(max(x, 0.5), 3.0))
            df[col] = df[col].round(2)  # Round to 2 decimal places
        
        # Save the enhanced dataframe
        # df.to_excel(output_file, index=False)
        # context.log(f"\nData with Pythagorean ratings saved to {output_file}")
        # context.log(f"All values are scaled to be between 0.5-3")
        

        # drop unwanted columns
        df = df.drop(['home_team_luck_factor'], axis=1)

        return df

    # Example usage
    df_pyth = add_pythagorean_ratings(
        df_bt, 
        num_recent_games=NUM_PREVIOUS_GAMES, 
        exponent=PYTHAGOREAN_EXPONENT, 
    )

    # Save the enhanced dataframe with only the xG columns
    # df_pyth.to_excel(OUTPUT_FILENAME, index=False)
    # context.log(f"Data with only xG values saved to {OUTPUT_FILENAME}")

    ###########################################################################################################################
    #################################     BAYESIAN MODEL   ####################################################################

    def _calculate_time_weights(match_indices, half_life=10):
        """
        Calculate time weights for matches based on their recency.
        
        Parameters:
        -----------
        match_indices : array-like
            Indices of matches in chronological order
        half_life : int, optional
            Number of matches after which the weight is halved (default: 10)
        
        Returns:
        --------
        numpy.array
            Array of weights, with more recent matches having higher weights
        """
        # Convert to numpy array for efficiency
        indices = np.array(match_indices)
        
        # Calculate the time difference (in match count) from the most recent match
        most_recent_idx = max(indices)
        match_age = most_recent_idx - indices
        
        # Calculate weights using exponential decay
        weights = np.exp(-np.log(2) * match_age / half_life)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights

    def _get_prior_parameters(df, prior_strength=1.0):
        """
        Calculate prior parameters for the Bayesian model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing match data
        prior_strength : float, optional
            Relative strength of the prior (higher values mean stronger prior)
        
        Returns:
        --------
        tuple
            (prior_attack, prior_defense, prior_home_advantage, prior_precision)
        """
        # Calculate league averages
        home_goals = df['home_goals'].mean()
        away_goals = df['away_goals'].mean()
        
        # Default values if data is insufficient
        if pd.isna(home_goals) or pd.isna(away_goals) or home_goals <= 0 or away_goals <= 0:
            home_goals = 1.5
            away_goals = 1.1
        
        # Prior for team attack and defense - set to 1.5 (middle of 0-3 scale)
        prior_attack = 1.5
        prior_defense = 1.5
        
        # Prior for home advantage - set to 1.1
        prior_home_advantage = 1.1
        
        # Prior precision (inverse variance)
        # Lower values = less certain prior (more weight on the data)
        # Higher values = more certain prior (less weight on the data)
        prior_precision = prior_strength / (np.std([home_goals, away_goals]) + 0.1)
        
        return prior_attack, prior_defense, prior_home_advantage, prior_precision

    def _update_bayesian_parameters(
        team, is_home, goals_for, goals_against,
        attack_params, defense_params, 
        opponent, home_advantage, prior_attack, prior_defense, prior_precision
    ):
        """
        Update Bayesian parameters for a team based on match results.
        
        Parameters:
        -----------
        team : str
            Team name
        is_home : bool
            Whether the team was playing at home
        goals_for : int
            Goals scored by the team
        goals_against : int
            Goals conceded by the team
        attack_params : dict
            Current attack parameters (alpha, beta) for all teams
        defense_params : dict
            Current defense parameters (alpha, beta) for all teams
        opponent : str
            Opponent team name
        home_advantage : float
            Home advantage factor
        prior_attack, prior_defense : float
            Prior means for attack and defense
        prior_precision : float
            Precision (inverse variance) of the prior
        
        Returns:
        --------
        tuple
            (updated_attack_params, updated_defense_params)
        """
        # Initialize parameters if not present
        if team not in attack_params:
            attack_params[team] = (prior_attack, prior_precision)
        if team not in defense_params:
            defense_params[team] = (prior_defense, prior_precision)
        if opponent not in attack_params:
            attack_params[opponent] = (prior_attack, prior_precision)
        if opponent not in defense_params:
            defense_params[opponent] = (prior_defense, prior_precision)
        
        # Get current parameters
        attack_alpha, attack_beta = attack_params[team]
        defense_alpha, defense_beta = defense_params[team]
        opp_attack_alpha, opp_attack_beta = attack_params[opponent]
        opp_defense_alpha, opp_defense_beta = defense_params[opponent]
        
        # Expected goals - Note the approach with defense parameter scaling
        if is_home:
            expected_goals_for = attack_alpha * ((3.0 - opp_defense_alpha) / 3.0) * home_advantage
            expected_goals_against = opp_attack_alpha * ((3.0 - defense_alpha) / 3.0)
        else:
            expected_goals_for = attack_alpha * ((3.0 - opp_defense_alpha) / 3.0)
            expected_goals_against = opp_attack_alpha * ((3.0 - defense_alpha) / 3.0) * home_advantage
        
        # Calculate likelihoods (simplification of Poisson likelihood)
        # Higher values indicate the current parameters explain the observed goals well
        attack_likelihood = poisson.pmf(goals_for, expected_goals_for) if expected_goals_for > 0 else 0.01
        defense_likelihood = poisson.pmf(goals_against, expected_goals_against) if expected_goals_against > 0 else 0.01
        
        # Update attack parameters using Bayesian update rule
        # Higher likelihood -> parameters move more toward observed data
        # Higher beta (prior precision) -> parameters stay closer to prior values
        attack_alpha_new = (attack_alpha * attack_beta + (goals_for / 2.0)) / (attack_beta + 0.5)
        attack_beta_new = attack_beta + attack_likelihood
        
        # Update defense parameters - for defense, higher values mean fewer goals conceded
        # We reward good defense with higher defensive ratings
        defense_adjustment = 3.0 / (goals_against + 1.0)  # Transforms goals against to a 0-3 scale value
        defense_alpha_new = (defense_alpha * defense_beta + defense_adjustment) / (defense_beta + 1)
        defense_beta_new = defense_beta + defense_likelihood
        
        # Update parameters with smoothing to prevent extreme values
        smooth_factor = 0.7  # Adjust for more/less responsiveness
        attack_alpha = (1 - smooth_factor) * attack_alpha + smooth_factor * attack_alpha_new
        attack_beta = (1 - smooth_factor) * attack_beta + smooth_factor * attack_beta_new
        defense_alpha = (1 - smooth_factor) * defense_alpha + smooth_factor * defense_alpha_new
        defense_beta = (1 - smooth_factor) * defense_beta + smooth_factor * defense_beta_new
        
        # Constrain parameters to 0-3 range for alpha (team strength)
        attack_alpha = min(max(attack_alpha, 0.0), 3.0)
        defense_alpha = min(max(defense_alpha, 0.0), 3.0)
        attack_beta = min(max(attack_beta, prior_precision * 0.5), prior_precision * 10)
        defense_beta = min(max(defense_beta, prior_precision * 0.5), prior_precision * 10)
        
        # Update parameter dictionaries
        attack_params[team] = (attack_alpha, attack_beta)
        defense_params[team] = (defense_alpha, defense_beta)
        
        return attack_params, defense_params

    def _calculate_expected_goals(home_team, away_team, attack_params, defense_params, home_advantage):
        """
        Calculate expected goals for a match using Bayesian parameters.
        
        Parameters:
        -----------
        home_team, away_team : str
            Team names
        attack_params, defense_params : dict
            Dictionaries containing attack and defense parameters for all teams
        home_advantage : float
            Home advantage factor
        
        Returns:
        --------
        tuple
            (expected_home_goals, expected_away_goals)
        """
        # Get parameters - default to middle value (1.5) if not available
        home_attack_alpha, _ = attack_params.get(home_team, (1.5, 1.0))
        home_defense_alpha, _ = defense_params.get(home_team, (1.5, 1.0))
        away_attack_alpha, _ = attack_params.get(away_team, (1.5, 1.0))
        away_defense_alpha, _ = defense_params.get(away_team, (1.5, 1.0))
        
        # For defense, higher values mean better defense (fewer goals conceded)
        # We need to convert defense rating to a factor that reduces expected goals
        home_defense_factor = (3.0 - away_defense_alpha) / 3.0  # Transform to a 0-1 scale
        away_defense_factor = (3.0 - home_defense_alpha) / 3.0  # Transform to a 0-1 scale
        
        # Calculate expected goals using more direct scaling from the 0-3 values
        # This approach ensures that higher attack values correspond to more goals
        # and higher defense values correspond to fewer goals conceded
        expected_home_goals = home_attack_alpha * away_defense_factor * home_advantage
        expected_away_goals = away_attack_alpha * home_defense_factor
        
        # Ensure expected goals are within the 0-3 range
        expected_home_goals = min(max(expected_home_goals, 0.0), 3.0)
        expected_away_goals = min(max(expected_away_goals, 0.0), 3.0)
        
        return expected_home_goals, expected_away_goals

    def _calculate_match_probabilities(expected_home_goals, expected_away_goals, max_goals=10):
        """
        Calculate match outcome probabilities based on expected goals.
        
        Parameters:
        -----------
        expected_home_goals, expected_away_goals : float
            Expected goals for home and away teams
        max_goals : int, optional
            Maximum number of goals to consider for each team
        
        Returns:
        --------
        tuple
            (home_win_prob, draw_prob, away_win_prob)
        """
        # Initialize counters
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        
        # Ensure positive expected goals
        expected_home_goals = max(expected_home_goals, 0.01)
        expected_away_goals = max(expected_away_goals, 0.01)
        
        # Calculate probabilities for each possible score
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Probability of this exact score
                home_pmf = poisson.pmf(home_goals, expected_home_goals)
                away_pmf = poisson.pmf(away_goals, expected_away_goals)
                score_prob = home_pmf * away_pmf
                
                # Add to the appropriate outcome
                if home_goals > away_goals:
                    home_win_prob += score_prob
                elif home_goals == away_goals:
                    draw_prob += score_prob
                else:
                    away_win_prob += score_prob
        
        # Normalize probabilities
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob > 0:
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
        else:
            home_win_prob = draw_prob = away_win_prob = 1/3
        
        return home_win_prob, draw_prob, away_win_prob

    def _extract_team_strengths(attack_params, defense_params):
        """
        Extract team strengths from Bayesian parameters and ensure they're within 0-3 range.
        
        Parameters:
        -----------
        attack_params, defense_params : dict
            Dictionaries containing attack and defense parameters for all teams
        
        Returns:
        --------
        tuple
            (attack_strengths, defense_strengths)
        """
        attack_strengths = {}
        defense_strengths = {}
        
        # Extract alpha values directly
        for team, (alpha, _) in attack_params.items():
            attack_strengths[team] = alpha
        
        for team, (alpha, _) in defense_params.items():
            defense_strengths[team] = alpha
        
        # Apply final clipping to ensure all values are in 0-3 range
        attack_strengths = {team: min(max(strength, 0.0), 3.0) for team, strength in attack_strengths.items()}
        defense_strengths = {team: min(max(strength, 0.0), 3.0) for team, strength in defense_strengths.items()}
        
        return attack_strengths, defense_strengths

    def add_bayesian_ratings(dataframe, num_recent_games=6, prior_strength=0.6, update_interval=6):
        """
        Add Bayesian hierarchical model ratings to the football dataset.
        All ratings and expected goals are scaled to the 0-3 range.
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file containing the football data
        num_recent_games : int, optional
            Number of recent games to consider for calculations (default: 6)
        output_file : str, optional
            Path to save the processed data (default: None, which appends '_bayesian' to original filename)
        prior_strength : float, optional
            Strength of the prior (default: 0.7, higher values give more weight to prior)
        update_interval : int, optional
            Number of matches after which to recalculate summaries (default: 5)
        
        Returns:
        --------
        pandas.DataFrame
            The enhanced dataframe with Bayesian model rating columns
        """
        # Set default output file name if not provided
        # if output_file is None:
        #     output_file = file_path.replace('.xlsx', '_bayesian_0_3.xlsx')
        
        # Load and process the data
        # df = load_and_process_data(file_path)
        df = dataframe

        if df is None:
            context.log("Failed to load data. Exiting.")
            return None
        
        # Create a sequential index for chronological ordering
        df = df.reset_index(drop=True)
        
        # Initialize rating columns
        bayesian_columns = [
            'home_team_attack_bayesian',
            'home_team_defense_bayesian',
            'away_team_attack_bayesian',
            'away_team_defense_bayesian',
            'home_advantage_bayesian',
            'home_xg_bayesian',
            'away_xg_bayesian',
        ]
        
        # Initialize all columns with default values - set to middle of 0-3 range (1.5)
        for col in bayesian_columns:
            if col.startswith('expected_'):
                df[col] = 1.5  # Default expected goals is 1.5 (middle of 0-3 range)
            elif col == 'home_advantage_bayesian':
                df[col] = 1.1  # Default home advantage is 1.1
            else:
                df[col] = 1.5  # Default strength is 1.5 on the 0-3 scale
        
        # Process each season separately
        seasons = sorted(df['season'].unique())
        for season in seasons:
            context.log(f"\nProcessing season: {season}")
            season_df = df[df['season'] == season].copy()
            season_indices = season_df.index.tolist()
            
            # Minimum matches needed before we start calculating
            min_matches_needed = num_recent_games
            
            # Initialize Bayesian parameters with default 1.5 strength (middle of 0-3 range)
            attack_params = {}  # (alpha, beta) for each team
            defense_params = {}  # (alpha, beta) for each team
            
            # Calculate prior parameters - using 1.1 for home advantage
            prior_attack, prior_defense, home_advantage, prior_precision = _get_prior_parameters(
                df, prior_strength=prior_strength
            )
            
            # Last update position for periodic recalculation
            last_update_position = -1
            
            # Track team strengths for this season
            attack_strengths = {}
            defense_strengths = {}
            
            # Process each match chronologically within the season
            for match_idx_position, match_idx in enumerate(season_indices):
                home_team = df.at[match_idx, 'home_team']
                away_team = df.at[match_idx, 'away_team']
                
                # Get all previous matches in this season up to this match
                prev_season_matches = season_df[season_df.index < match_idx]
                
                # Check if we have enough matches to fit the model
                if len(prev_season_matches) >= min_matches_needed:
                    # For each team, find recent matches to update their parameters
                    all_teams = set(prev_season_matches['home_team'].unique()) | set(prev_season_matches['away_team'].unique())
                    
                    # Skip if either team hasn't played yet
                    if home_team not in all_teams or away_team not in all_teams:
                        continue
                    
                    # Filter recent matches for relevant teams
                    team_recent_matches = {}
                    
                    for team in [home_team, away_team]:
                        team_matches = prev_season_matches[(prev_season_matches['home_team'] == team) | 
                                                        (prev_season_matches['away_team'] == team)]
                        
                        # Sort by index (chronological order) and take the most recent matches
                        if len(team_matches) > 0:
                            team_recent_matches[team] = team_matches.sort_index(ascending=False).head(num_recent_games)
                    
                    # Process each team's recent matches to update their parameters
                    for team, matches in team_recent_matches.items():
                        for _, match in matches.iterrows():
                            match_home_team = match['home_team']
                            match_away_team = match['away_team']
                            
                            # Update parameters based on match outcome
                            if match_home_team == team:
                                attack_params, defense_params = _update_bayesian_parameters(
                                    team, True, match['home_goals'], match['away_goals'],
                                    attack_params, defense_params, 
                                    match_away_team, home_advantage, 
                                    prior_attack, prior_defense, prior_precision
                                )
                            else:  # Away team
                                attack_params, defense_params = _update_bayesian_parameters(
                                    team, False, match['away_goals'], match['home_goals'],
                                    attack_params, defense_params, 
                                    match_home_team, home_advantage, 
                                    prior_attack, prior_defense, prior_precision
                                )
                    
                    # Periodically recalculate team strengths from parameters
                    if match_idx_position - last_update_position >= update_interval:
                        last_update_position = match_idx_position
                        attack_strengths, defense_strengths = _extract_team_strengths(attack_params, defense_params)
                        
                        # context.log progress
                        if match_idx_position % 100 == 0:
                            context.log(f"  Updated model at match {match_idx_position+1}/{len(season_indices)}")
                    
                    # Calculate expected goals for this match
                    expected_home_goals, expected_away_goals = _calculate_expected_goals(
                        home_team, away_team, attack_params, defense_params, home_advantage
                    )
                    
                    # Update ratings for current match - all rounded to 2 decimal places
                    df.at[match_idx, 'home_team_attack_bayesian'] = np.round(attack_strengths.get(home_team, 1.5), 2)
                    df.at[match_idx, 'home_team_defense_bayesian'] = np.round(defense_strengths.get(home_team, 1.5), 2)
                    df.at[match_idx, 'away_team_attack_bayesian'] = np.round(attack_strengths.get(away_team, 1.5), 2)
                    df.at[match_idx, 'away_team_defense_bayesian'] = np.round(defense_strengths.get(away_team, 1.5), 2)
                    df.at[match_idx, 'home_advantage_bayesian'] = np.round(home_advantage, 2)
                    df.at[match_idx, 'home_xg_bayesian'] = np.round(expected_home_goals, 2)
                    df.at[match_idx, 'away_xg_bayesian'] = np.round(expected_away_goals, 2)
        
        # Fill any remaining NaN values with default values
        for col in bayesian_columns:
            if col.startswith('expected_'):
                df[col] = df[col].fillna(1.5)
            elif col == 'home_advantage_bayesian':
                df[col] = df[col].fillna(1.1)
            else:
                df[col] = df[col].fillna(1.5)
            
            # Ensure all values are rounded to 2 decimal places
            if col in df.columns:
                df[col] = df[col].round(2)
            
            # Final safety check to ensure all values are within required 0-3 range
            if col != 'home_advantage_bayesian':  # Skip home advantage which is fixed at 1.1
                df[col] = df[col].apply(lambda x: min(max(x, 0.0), 3.0))
        

        df = df.drop(['home_team_attack_bayesian', 'home_team_defense_bayesian', 'away_team_attack_bayesian', 'away_team_defense_bayesian', 'home_advantage_bayesian'], axis=1)

        # Save the enhanced dataframe
        # df.to_excel(output_file, index=False)
        # context.log(f"\nData with Bayesian hierarchical model ratings saved to {output_file}")
        # context.log(f"All values are scaled to be between 0-3, and home advantage is set to 1.1")
        
        return df

    # Example usage
    df_bayesian = add_bayesian_ratings(
        df_pyth, 
        num_recent_games=NUM_PREVIOUS_GAMES, 
        prior_strength=PRIOR_STRENGTH,
        update_interval=UPDATE_INTERVAL
    )

    # Save the enhanced dataframe with only the xG columns
    # df_bayesian.to_excel(OUTPUT_FILENAME, index=False)
    # context.log(f"Data with only xG values saved to {OUTPUT_FILENAME}")


    ###########################################################################################################################
    #################################     TIME WEIGHTED REGRESSION   ##########################################################

    def _calculate_time_weights(match_indices, half_life=6):
        """
        Calculate time weights for matches based on their recency.
        
        Parameters:
        -----------
        match_indices : array-like
            Indices of matches in chronological order
        half_life : int, optional
            Number of matches after which the weight is halved (default: 10)
        
        Returns:
        --------
        numpy.array
            Array of weights, with more recent matches having higher weights
        """
        # Convert to numpy array for efficiency
        indices = np.array(match_indices)
        
        # Calculate the time difference (in match count) from the most recent match
        most_recent_idx = max(indices)
        match_age = most_recent_idx - indices
        
        # Calculate weights using exponential decay
        weights = np.exp(-np.log(2) * match_age / half_life)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights

    def _prepare_home_xg_regression_data(matches_df):
        """
        Prepare data for time-weighted regression for home expected goals with
        robust handling to prevent singular matrix issues.
        
        Parameters:
        -----------
        matches_df : pandas.DataFrame
            DataFrame containing match data with required columns
        
        Returns:
        --------
        tuple
            (X, y) where X is the feature matrix and y is the target vector,
            or (None, None) if insufficient valid data
        """
        # Select only matches with valid home_xg_odds (not 0)
        valid_matches = matches_df[matches_df['home_xg_odds'] > 0].copy()
        
        if len(valid_matches) == 0:
            context.log("  No valid matches with non-zero home_xg_odds")
            return None, None
        
        # Check for missing values and replace with mean if any exist
        for col in ['home_team_goals_scored_average', 'away_team_goals_conceded_average']:
            if col not in valid_matches.columns:
                context.log(f"  Missing required column: {col}")
                return None, None
                
            if valid_matches[col].isna().any():
                col_mean = valid_matches[col].mean()
                context.log(f"  Filling {valid_matches[col].isna().sum()} NA values in {col} with mean: {col_mean:.3f}")
                valid_matches[col] = valid_matches[col].fillna(col_mean)
        
        # Check for zero variance in either feature
        for col in ['home_team_goals_scored_average', 'away_team_goals_conceded_average']:
            if valid_matches[col].std() < 1e-6:  # Almost zero variance
                context.log(f"  Warning: Near-zero variance in {col}, adding small random noise")
                valid_matches[col] = valid_matches[col] + np.random.normal(0, 0.01, len(valid_matches))
        
        # Check for perfect correlation between features (causes singularity)
        if len(valid_matches) > 1:
            correlation = np.corrcoef(
                valid_matches['home_team_goals_scored_average'], 
                valid_matches['away_team_goals_conceded_average']
            )[0, 1]
            
            if abs(correlation) > 0.997:  # Very high correlation
                context.log(f"  Warning: Features are highly correlated (r={correlation:.4f}), adding noise")
                # Add small random noise to one feature
                valid_matches['home_team_goals_scored_average'] += np.random.normal(0, 0.01, len(valid_matches))
        
        # Ensure all values are valid numbers
        for col in ['home_team_goals_scored_average', 'away_team_goals_conceded_average', 'home_xg_odds']:
            # Replace any remaining invalid values (inf, -inf, etc.)
            invalid_mask = ~np.isfinite(valid_matches[col])
            if invalid_mask.any():
                context.log(f"  Replacing {invalid_mask.sum()} invalid values in {col}")
                valid_matches.loc[invalid_mask, col] = valid_matches[col].mean()
        
        # Check if we have enough variation in target variable
        if valid_matches['home_xg_odds'].std() < 0.05:
            context.log("  Warning: Very low variance in target variable (home_xg_odds)")
            # Add tiny noise to target to avoid perfect fit issues
            valid_matches['home_xg_odds'] += np.random.normal(0, 0.005, len(valid_matches))
        
        # Create feature matrix and target vector
        X = valid_matches[['home_team_goals_scored_average', 'away_team_goals_conceded_average']].values.astype(float)
        y = valid_matches['home_xg_odds'].values.astype(float)
        
        # Final validity check
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            context.log("  Error: Non-finite values remain in data after preprocessing")
            return None, None
        
        # Debug info
        context.log(f"  Prepared home xG regression data: {len(X)} samples, X shape: {X.shape}")
        
        return X, y

    def _prepare_away_xg_regression_data(matches_df):
        """
        Prepare data for time-weighted regression for away expected goals with
        robust handling to prevent singular matrix issues.
        
        Parameters:
        -----------
        matches_df : pandas.DataFrame
            DataFrame containing match data with required columns
        
        Returns:
        --------
        tuple
            (X, y) where X is the feature matrix and y is the target vector,
            or (None, None) if insufficient valid data
        """
        # Select only matches with valid away_xg_odds (not 0)
        valid_matches = matches_df[matches_df['away_xg_odds'] > 0].copy()
        
        if len(valid_matches) == 0:
            context.log("  No valid matches with non-zero away_xg_odds")
            return None, None
        
        # Check for missing values and replace with mean if any exist
        for col in ['away_team_goals_scored_average', 'home_team_goals_conceded_average']:
            if col not in valid_matches.columns:
                context.log(f"  Missing required column: {col}")
                return None, None
                
            if valid_matches[col].isna().any():
                col_mean = valid_matches[col].mean()
                context.log(f"  Filling {valid_matches[col].isna().sum()} NA values in {col} with mean: {col_mean:.3f}")
                valid_matches[col] = valid_matches[col].fillna(col_mean)
        
        # Check for zero variance in either feature
        for col in ['away_team_goals_scored_average', 'home_team_goals_conceded_average']:
            if valid_matches[col].std() < 1e-6:  # Almost zero variance
                context.log(f"  Warning: Near-zero variance in {col}, adding small random noise")
                valid_matches[col] = valid_matches[col] + np.random.normal(0, 0.01, len(valid_matches))
        
        # Check for perfect correlation between features (causes singularity)
        if len(valid_matches) > 1:
            correlation = np.corrcoef(
                valid_matches['away_team_goals_scored_average'], 
                valid_matches['home_team_goals_conceded_average']
            )[0, 1]
            
            if abs(correlation) > 0.997:  # Very high correlation
                context.log(f"  Warning: Features are highly correlated (r={correlation:.4f}), adding noise")
                # Add small random noise to one feature
                valid_matches['away_team_goals_scored_average'] += np.random.normal(0, 0.01, len(valid_matches))
        
        # Ensure all values are valid numbers
        for col in ['away_team_goals_scored_average', 'home_team_goals_conceded_average', 'away_xg_odds']:
            # Replace any remaining invalid values (inf, -inf, etc.)
            invalid_mask = ~np.isfinite(valid_matches[col])
            if invalid_mask.any():
                context.log(f"  Replacing {invalid_mask.sum()} invalid values in {col}")
                valid_matches.loc[invalid_mask, col] = valid_matches[col].mean()
        
        # Check if we have enough variation in target variable
        if valid_matches['away_xg_odds'].std() < 0.05:
            context.log("  Warning: Very low variance in target variable (away_xg_odds)")
            # Add tiny noise to target to avoid perfect fit issues
            valid_matches['away_xg_odds'] += np.random.normal(0, 0.005, len(valid_matches))
        
        # Create feature matrix and target vector
        X = valid_matches[['away_team_goals_scored_average', 'home_team_goals_conceded_average']].values.astype(float)
        y = valid_matches['away_xg_odds'].values.astype(float)
        
        # Final validity check
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            context.log("  Error: Non-finite values remain in data after preprocessing")
            return None, None
        
        # Debug info
        context.log(f"  Prepared away xG regression data: {len(X)} samples, X shape: {X.shape}")
        
        return X, y

    def _fit_time_weighted_regression(X, y, weights):
        """
        Fit a time-weighted linear regression model using statsmodels.
        
        Parameters:
        -----------
        X : numpy.array
            Feature matrix
        y : numpy.array
            Target vector
        weights : numpy.array
            Sample weights (higher weights for more recent matches)
        
        Returns:
        --------
        dict
            Fitted regression model with predict method
        """
        try:
            # Add constant (intercept) term
            X_with_const = sm.add_constant(X)
            
            # Fit weighted least squares model
            wls_model = sm.WLS(y, X_with_const, weights=weights)
            results = wls_model.fit()
            
            # Create a model dict with a predict method that correctly adds the constant
            def predict_func(X_new):
                # Make sure X_new is 2D
                if X_new.ndim == 1:
                    X_new = X_new.reshape(1, -1)
                    
                # Add constant column if not already present
                X_new_with_const = sm.add_constant(X_new)
                
                # Handle the case where X_new has only one row and add_constant didn't work as expected
                if X_new_with_const.shape[1] != X_with_const.shape[1]:
                    # Manually add constant
                    X_new_with_const = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
                    
                return results.predict(X_new_with_const)
            
            model = {
                'intercept_': results.params[0],
                'coef_': results.params[1:],
                'predict': predict_func
            }
            
            return model
        
        except Exception as e:
            context.log(f"  Statsmodels regression error: {e}")
            
            # Fallback to simple mean-based prediction if regression fails
            mean_y = np.average(y, weights=weights)
            
            # Return a simple model that always predicts the weighted mean
            fallback_model = {
                'intercept_': mean_y,
                'coef_': np.zeros(X.shape[1]),
                'predict': lambda X_new: np.full(X_new.shape[0] if X_new.ndim > 1 else 1, mean_y)
            }
            
            context.log(f"  Using fallback mean-based model (mean={mean_y:.3f})")
            return fallback_model

    def predict_xg_with_model(model, home_goals_scored=None, away_goals_conceded=None, 
                            away_goals_scored=None, home_goals_conceded=None):
        """
        Predict expected goals using a fitted model.
        """
        if model is None:
            return 0.0
        
        # For home model
        if home_goals_scored is not None and away_goals_conceded is not None:
            X_pred = np.array([home_goals_scored, away_goals_conceded]).reshape(1, 2)
            pred = model['predict'](X_pred)
            return max(0, pred[0])
        
        # For away model
        elif away_goals_scored is not None and home_goals_conceded is not None:
            X_pred = np.array([away_goals_scored, home_goals_conceded]).reshape(1, 2)
            pred = model['predict'](X_pred)
            return max(0, pred[0])
        
        return 0.0

    def add_time_weighted_xg_regression(dataframe, num_recent_games=6, time_decay_half_life=6):
        """
        Add time-weighted regression model for expected goals to the football dataset.
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file containing the football data
        num_recent_games : int, optional
            Number of recent games to consider for calculations (default: 6)
        time_decay_half_life : int, optional
            Number of matches after which the weight is halved (default: 10)
        output_file : str, optional
            Path to save the processed data (default: None, which appends '_xg_twr' to original filename)
        
        Returns:
        --------
        pandas.DataFrame
            The enhanced dataframe with time-weighted regression expected goals columns
        """
        # Set default output file name if not provided
        # if output_file is None:
        #     output_file = file_path.replace('.xlsx', '_xg_twr.xlsx')
        
        # Load and process the data
        # df = load_and_process_data(file_path)
        df = dataframe
        
        if df is None:
            context.log("Failed to load data. Exiting.")
            return None
        
        # Create a sequential index for chronological ordering
        df = df.reset_index(drop=True)
        
        # Initialize expected goals columns
        twr_columns = [
            'home_xg_twr',
            'away_xg_twr',
        ]
        
        # Initialize all columns with default values
        for col in twr_columns:
            df[col] = 0.0
        
        # Process each season separately
        seasons = sorted(df['season'].unique())
        for i, season in enumerate(seasons):
            context.log(f"\nProcessing season: {season}")
            season_df = df[df['season'] == season].copy()
            season_indices = season_df.index.tolist()
            
            # Adjust minimum matches needed based on season position
            if i == 0:
                # First season - start calculating earlier
                min_matches_needed = max(5, num_recent_games-1)
            else:
                # Subsequent seasons - standard threshold
                min_matches_needed = max(5, num_recent_games-1)
            
            # Store models for this season
            home_xg_model = None
            away_xg_model = None
            
            # Process each match chronologically within the season
            for match_idx_position, match_idx in enumerate(season_indices):
                # Current match data
                home_team = df.at[match_idx, 'home_team']
                away_team = df.at[match_idx, 'away_team']
                home_goals_scored_avg = df.at[match_idx, 'home_team_goals_scored_average']
                away_goals_conceded_avg = df.at[match_idx, 'away_team_goals_conceded_average']
                away_goals_scored_avg = df.at[match_idx, 'away_team_goals_scored_average']
                home_goals_conceded_avg = df.at[match_idx, 'home_team_goals_conceded_average']
                
                # Get all previous matches in this season up to this match
                prev_season_matches = season_df[season_df.index < match_idx]
                
                # Check if we have enough matches to fit the model
                if len(prev_season_matches) >= min_matches_needed:
                    # Only refit the model periodically to save computation time
                    # if (match_idx_position % 10 == 0) or (home_xg_model is None or away_xg_model is None):
                    if (match_idx_position % time_decay_half_life == 0) or (home_xg_model is None or away_xg_model is None):
                        # Calculate time weights for previous matches
                        weights = _calculate_time_weights(prev_season_matches.index, time_decay_half_life)
                        
                        # Prepare data for home expected goals regression
                        X_home, y_home = _prepare_home_xg_regression_data(prev_season_matches)
                        
                        # Fit home expected goals model if data is available
                        if X_home is not None and y_home is not None and len(X_home) >= min_matches_needed:
                            try:
                                home_xg_model = _fit_time_weighted_regression(X_home, y_home, weights[:len(X_home)])
                                # r2_home = home_xg_model.score(X_home, y_home)
                                # context.log(f"  Home xG model R² = {r2_home:.4f} (fitted on {len(X_home)} matches)")
                            except Exception as e:
                                context.log(f"  Error fitting home xG model: {e}")
                        
                        # Prepare data for away expected goals regression
                        X_away, y_away = _prepare_away_xg_regression_data(prev_season_matches)
                        
                        # Fit away expected goals model if data is available
                        if X_away is not None and y_away is not None and len(X_away) >= min_matches_needed:
                            try:
                                away_xg_model = _fit_time_weighted_regression(X_away, y_away, weights[:len(X_away)])
                                # r2_away = away_xg_model.score(X_away, y_away)
                                # context.log(f"  Away xG model R² = {r2_away:.4f} (fitted on {len(X_away)} matches)")
                            except Exception as e:
                                context.log(f"  Error fitting away xG model: {e}")
                
                # Predict expected goals using the models
                home_xg_pred = predict_xg_with_model(
                    home_xg_model, 
                    home_goals_scored=home_goals_scored_avg, 
                    away_goals_conceded=away_goals_conceded_avg
                )
                
                away_xg_pred = predict_xg_with_model(
                    away_xg_model, 
                    away_goals_scored=away_goals_scored_avg, 
                    home_goals_conceded=home_goals_conceded_avg
                )
                
                # Store predictions in the dataframe
                df.at[match_idx, 'home_xg_twr'] = np.round(home_xg_pred, 2)
                df.at[match_idx, 'away_xg_twr'] = np.round(away_xg_pred, 2)
                
                # context.log progress periodically
                # if match_idx_position % 50 == 0 or match_idx_position == len(season_indices) - 1:
                #     context.log(f"  Processing match {match_idx_position+1}/{len(season_indices)}: {home_team} vs {away_team}")
                #     context.log(f"  Home xG: {home_xg_pred:.2f}, Away xG: {away_xg_pred:.2f}")
        
        # Save the enhanced dataframe
        # df.to_excel(output_file, index=False)
        # context.log(f"\nData with time-weighted regression expected goals saved to {output_file}")
        
        return df

    # Example usage
    df_twr = add_time_weighted_xg_regression(
        df_bayesian, 
        num_recent_games=NUM_PREVIOUS_GAMES, 
        time_decay_half_life=HALF_LIFE, 
    )

    df_twr = formatDataframe(df_twr)

    # Save the enhanced dataframe with only the xG columns
    # df_twr.to_excel(OUTPUT_FILENAME, index=False)
    # context.log(f"Data with only xG values saved to {OUTPUT_FILENAME}")

    save_to_appwrite_storage(df_twr, STORAGE_ID, file_id="nfdStats", client=client)

    return context.res.empty()
