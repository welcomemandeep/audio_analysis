import nltk.corpus
import nltk
nltk.download('punkt')
nltk.download('nps_chat')

# Inspired from https://github.com/kartikn27/nlp-question-detection

class IsQuestion():
    def __init__(self):
        posts = self.__get_posts()
        feature_set = self.__get_feature_set(posts)
        self.classifier = self.__perform_classification(feature_set)


    def __get_posts(self):
        return nltk.corpus.nps_chat.xml_posts()

    def __get_feature_set(self, posts):
        feature_list = []
        for post in posts:
            post_text = post.text
            features = {}
            words = nltk.word_tokenize(post_text)
            for word in words:
                features['contains({})'.format(word.lower())] = True
            feature_list.append((features, post.get('class')))
        return feature_list


    def __perform_classification(self, feature_set):
        training_size = int(len(feature_set) * 0.1)
        train_set, test_set = feature_set[training_size:], feature_set[:training_size]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print('Accuracy is : ', nltk.classify.accuracy(classifier, test_set))
        return classifier

    def __get_question_words_set(self):
        question_word_list = ['what', 'where', 'when', 'how', 'why', 'did', 'do', 'does', 'have', 'has', 'am', 'is',
        'are', 'can', 'could', 'may', 'would', 'will', 'should', "didn't", "doesn't", "haven't", "isn't",
         "aren't", "can't", "couldn't", "wouldn't", "won't", "shouldn't", '?']
        return set(question_word_list)


    def predict_question(self, text):
        words = nltk.word_tokenize(text.lower())
        if self.__get_question_words_set().intersection(words) == False:
            return 0
        if '?' in text:
            return 1

        features = {}
        for word in words:
            features['contains({})'.format(word.lower())] = True

        prediction_result = self.classifier.classify(features)
        if prediction_result == 'whQuestion' or prediction_result == 'ynQuestion':
            return 1
        return 0

