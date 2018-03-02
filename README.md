# Term Frequency Algorithm
## Basic Term Frequency Algorithm
### How It Works
#### Preprocessing
The algorithm cycles through the files in the directory containing the articles that are of the same topic, one by one. As it works on each document, it begins by cleaning the texts, removing punctuation, replacing all new line characters with a space and casting all characters to lower case, so that they are not considered distinct, regardless of the case. Then the numbers in the text are removed, because numbers carry very little context on their own. The sentences are then split into words, and the stop words (based on SpaCy) are removed. We also take this opportunity to track the length of document, saving it in a data structure of document lengths.

#### Creating the Dictionary
By now, the sentences have been broken down into its constituent words. We go through these words, counting them. As we encounter new words, we add them to our dictionary of words; otherwise, we increment the total number of times the word has been seen thus far. This dictionary of words keeps a separate entry for each document, so even if we encounter the same word in a different document, it is not added to the total that we have thus far. We also keep track of the word we have seen most frequently here, to be used for calculating the `tf` values

#### Calculating `tf`
Once we have constructed our dictionary for the document, we calculate the `tf` value for each word. We use the augmented term frequency formula, which was formulated to prevent a bias towards longer documents, e.g. raw frequency divided by the raw frequency of the most occurring term in the document.

#### Using Structural Information
We then try to make use of the structural information that we have. As these are news articles, which are based on the inverted pyramid structure, we assume that the most important information is placed at the start of the article. So we split all the sentences in the article into three large sections, giving each section of a different multiplier.

#### Calculating Sentence Scores
We then split our sentences into words, and look up its corresponding `tf` value in the dictionary that we had created earlier. We sum up the `tf` values for every word in that sentence that has had a value calculated, i.e. is not a stop word, then we find the average `tf` score for the sentence. However, here, if the sentence is shorter than 70 characters then we begin penalizing the score; this is to ensure that the algorithm does not only favour short sentences (as short sentences have far fewer stop words, and we are averaging the `tf` value by the length of the sentence, which includes stop words). We then multiply this score by a multiplier, depending on which section the sentence lies in. Every sentence that lies in the first 1/3 of the document has its average score multiplied by 1.5, while the ones in the second 1/3 are given a multiplier of 1.25. We then normalize the resulting `tf` by the length of the document, guaranteeing that there are no biases caused by differing article lengths. These score calculations are appended to a list.

## Term Frequency-Inverse Document Frequency
### How It Works
#### Calculating `tf`
We only execute this part of the algorithm after we have established our dictionary of words, and have calculated the `tf` values for all the words. At this point, we also have a list containing the scores for every single sentence.

#### Calculating `idf`
We cycle through all the words that we have stored in our dictionary of words, calculating the `idf` value for each word, using the formula.

#### Calculating `tf-idf`
We then cycle through all our documents again, calculating the `tf-idf` score. Likewise to the calculations for `tf`, we go through every word in every sentence, adding up its `tf-idf` score, which is calculated by multiplying `tf` by `idf`. We then average it by the length of the sentence and, again, calculate its value after the multiplier, depending on which section of the document that the sentence lies in. These details are added to the list of scores that we have already computed.

## Reconstructing the Summary
We use the dynamic programming solution to the 0-1 Knapsack problem to help us reconstruct the summary. We memoize and choose the sentences with the highest 'value', i.e. the highest `tf` or `tf-idf` score for its given length, trying to fit it into the summary, according to the word count limitation that we have been given as user input.

Once the sentences have been chosen, they are sorted by order of appearance, so that the sentences flow relatively coherently, maintaining the structure of the content of the original documents.

### Report Notes
- The weighting given to the different parts of the document affect the results given by the summary
- Start penalizing any sentence that is less than half a Tweet long to prevent very short sentences that convey very little context from being chosen.
- Numbers have been removed because they carry very little context on their own.

70 characters is just good enough for a title.
https://moz.com/blog/title-tags-is-70-characters-the-best-practice-whiteboard-friday

Well, recently people have been doing some experiments to see just how many characters Google will index within a title tag. For years, we thought it was 70s. It's fluctuated. But recent experiments have shown that Google will index anywhere between 150, one person even showed that they will index over 1,000 characters, and I will link to these experiments in the post. But does this mean that you should use all of those characters to your advantage? Can you use them to your advantage? Well, I got really curious about this. So I decided to perform some experiments here on the SEOmoz blog with super long title tags. We're talking extreme title tags, like 200 characters long, 250 characters long, just blew them out of the water just to see what would happen.