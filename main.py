import pickle
import warnings

warnings.filterwarnings("ignore")


def main():
    with open('models/sentiment_classifier_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('models/sentiment_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)

    query = input('Phrase: ')
    query = [query]
    processed_query = vectorizer.transform(query)

    result = (clf.predict(processed_query))
    result = str(result)
    print(f'I think the person is being {result[2:-2].lower()}')


main()
