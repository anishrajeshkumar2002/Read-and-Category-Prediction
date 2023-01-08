# Read-and-Category-Prediction

# Objective
Read prediction: Predict given a (user,book) pair from ‘pairs Read.csv’ whether the user
would read the book (0 or 1). Accuracy will be measured in terms of the categorization accuracy (fraction
of correct predictions). The test set has been constructed such that exactly 50% of the pairs correspond
to read books and the other 50% do not.

Category prediction: Predict the category of a book from a review. Five categories are used
for this task, which can be seen in the baseline program, namely Children’s, Comics/Graphic Novels,
Fantasy, Mystery/Thriller, and Romance. Performance will be measured in terms of the fraction of
correct classifications.

# Process
Read Predictions: For this I looped through every book the user has read generated jaccard similarity between the books the user has read and the book 
for which we are trying to predict if the user has read or not.Afterwards I take the highest jaccard similarity from all the books the user has read and 
if it is above the threshold of 0.012 I assign a 1 in a feature vector which I am building to use for logistic regression. If it is below the threshold I
assign a 0 in the feature vector. For the second feature in the feature vector I assign a 1 if the book we are trying to compare has more than 27 users who 
have read/wrote a review about it and 0 if it has less than that and put that in the feature vector as well. I then split the feature vector and all of the 
true values/Y set into a train/test split and run logistic regression to get a predictions vector then test the predictions against the Y test set to get an accuracy.


Category Predictions:I first categorize all the data by word count by putting them in pairs such that (x,y) where x is the amount of times which the word 
appeared in the dataset and y is the word itself and store it in the vector. I then filter out all the stopwords and then store everything in a sparse matrix 
of size 100,000 rows by  60,000 columns because after much testing these dimensions gave me the best prediction accuracy. I then split the matrix into a train 
and validation set and trained a multinomial bayes classifier model and used it to predict the categories of books.

# Results
75.02% accuracy on Category Prediction
75.82% accuracy on Read Prediction
