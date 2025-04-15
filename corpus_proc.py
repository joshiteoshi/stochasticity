import spacy
from pronouncing import stresses, phones_for_word, stresses_for_word, search_stresses
from copy import deepcopy

import gzip, json, csv, heapq

def save_dict(dt: dict, out_path: str):
    cp = deepcopy(dt)
    for _, val in cp.items():
        val["labels"] = list(val["labels"])

    with open(out_path, 'w') as fp:
        json.dump(cp, fp)

def generate_dictionary(corpus_zip, min_samples, out_path: str="corpus/dictionary.json", save_per: int=100000, update_per:int=10000):
    """
    generates a dictionary of words, stresses, and possible labels from a corpus.
    saves to a json file.
    """
    nlp = spacy.load("en_core_web_sm")
    dictionary = dict()

    counter = 0

    for line in corpus_zip:
        line = json.loads(line.strip().lower())['s']
        line = line.replace(".", "").replace(",", "").replace("(", "").replace(")", "")

        doc = nlp(line)

        for token in doc:
            if token.text in dictionary:
                dictionary[token.text]["labels"].add(token.pos_)

            else:
                stresses = stresses_for_word(token.text)
                if stresses == []:
                    continue
                
                if token.text[0] == '\'':
                    stresses = []

                labels = set()
                labels.add(token.pos_)
                
                dictionary[token.text] = {"labels": labels,
                                        "stress": stresses}
        
        counter += 1
        if counter % update_per == 0:
            print(str(counter) + " done")

        if counter >= min_samples:
            save_dict(dictionary, out_path)
            break

        if counter % save_per == 0:
            save_dict(dictionary, out_path)

def generate_dataset(corpus_zip, min_samples: int=1000000, out_path: str="corpus/stress.csv"):
    nlp = spacy.load("en_core_web_sm")
    fp = open(out_path, 'w')
    fields = ['sample', 'stress', 'duration', 'labels']
    writer = csv.DictWriter(fp, fields)
    writer.writeheader()

    sample_idx = 0

    for line in corpus_zip:
        line = json.loads(line.strip().lower())['s']
        line = line.replace(".", "").replace(",", "").replace("(", "").replace(")", "")

        doc = nlp(line)

        if any([token.text[0] == '\'' for token in doc]):
            continue

        else:
            stresses = [[]]
            durations = [[]]
            for token in doc:
                word_stress = list(set(stresses_for_word(token.text)))
                temp_stress = []
                temp_duration = []

                for stress in word_stress:
                    for pattern in stresses:
                        temp_stress.append(pattern + [stress])
                        
                for stress in word_stress:
                    for duration in durations:
                        temp_duration.append(duration + [len(stress)])

                stresses = temp_stress
                durations = temp_duration

            stresses = ["".join(pattern) for pattern in stresses]

            labels = " ".join([token.pos_ for token in doc])

            for pattern, duration in zip(stresses, durations):
                writer.writerow({'sample': sample_idx, 'stress': pattern, 'duration': duration, 'labels': labels})
                sample_idx += 1
        
        if sample_idx > min_samples:
            break
        
    fp.close()