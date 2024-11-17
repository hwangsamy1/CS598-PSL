# https://campuswire.com/c/GB46E5679/feed/823

import pandas as pd

def wmae_eval():
    # Load the test dataset that contains actual labels for evaluation
    test_with_label = pd.read_csv('./Proj2_Data/test_with_label.csv')
    print(f"test_with_label.csv shape: {test_with_label.shape}")
    print()

    # Define the number of folds for cross-validation
    num_folds = 10

    # Initialize a list to store the Weighted Mean Absolute Errors (WMAE) for each fold
    wae = []

    # Iterate over each fold
    for i in range(num_folds):
        # Construct the file path for the test data of the current fold and load it
        file_path = f'./Proj2_Data/fold_{i + 1}/test.csv'
        test = pd.read_csv(file_path)
        print(f"test.csv shape: {test.shape}")

        # Drop the 'IsHoliday' column as it is not needed for merging
        test = test.drop(columns=['IsHoliday']).merge(test_with_label, on=['Date', 'Store', 'Dept'])
        print(f"merge test and test_with_label shape: {test.shape}")

        # Construct the file path for the predictions of the current fold and load it
        file_path = f'./Proj2_Data/fold_{i + 1}/mypred.csv'
        test_pred = pd.read_csv(file_path)
        print(f"mypred.csv shape: {test_pred.shape}")

        # Drop the 'IsHoliday' column from predictions as well
        test_pred = test_pred.drop(columns=['IsHoliday'])

        # Merge the test data with the predictions on the relevant columns
        new_test = test.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')
        print(f"merge test_pred and new_test shape: {new_test.shape}")

        # Extract the actual weekly sales and the predicted weekly sales
        actuals = new_test['Weekly_Sales']
        preds = new_test['Weekly_Pred']
        print(f"actuals shape: {actuals.shape}")
        print(f"preds shape: {preds.shape}")

        # Create weights: assign a weight of 5 for holidays and 1 otherwise
        weights = new_test['IsHoliday'].apply(lambda x: 5 if x else 1)

        # Calculate the weighted mean absolute error and append it to the list
        # Sum of weighted absolute differences divided by the sum of weights
        wae.append(sum(weights * abs(actuals - preds)) / sum(weights))

        print()

    # Return the list of WMAE values for each fold
    return wae