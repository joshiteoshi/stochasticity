import spacy
from pronouncing import stresses_for_word
from copy import deepcopy

import json, csv, sqlite3

class WordDictionary:
    """
    dictionary class to mediate database access.
    i've never used sqlite before so idk what the typical way to do this is.
    """
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()
    
    def get_words_from(self, label: str, stress: str):
        """gets all words satisfying a label and stress pattern."""
        query = """
        SELECT DISTINCT ws.word
        FROM word_stress ws
        JOIN word_labels wl ON ws.word = wl.word
        WHERE ws.stress = ? AND wl.label = ?
        """
        return self.conn.execute(query, (stress, label)).fetchall()


def generate_dictionary(corpus_zip, min_samples, out_path: str="corpus/dictionary.db", save_per: int=100000, update_per:int=10000):
    """
    generates a dictionary of words, stresses, and possible labels from a corpus.
    saves to a json file.
    """
    conn, cur = make_database(out_path)
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
            save_dict(dictionary, conn, cur)
            break

        if counter % save_per == 0:
            save_dict(dictionary, conn, cur)
            conn.close()


def generate_dataset(corpus_zip, min_samples: int=1000000, out_path: str="corpus/stress.csv"):
    """
    generates a csv of data points associating stress patterns, durations, and labels.
    """
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


# define the structure of our dictionary database
dict_database_schema = """
DROP TABLE IF EXISTS words;
DROP TABLE IF EXISTS word_stress;
DROP TABLE IF EXISTS word_labels;

CREATE TABLE words (
    word TEXT PRIMARY KEY
);

CREATE TABLE word_stress (
    word TEXT,
    stress TEXT,
    FOREIGN KEY(word) REFERENCES words(word),
    UNIQUE(word, stress)
);

CREATE INDEX idx_word_stress_stress ON word_stress(stress);

CREATE TABLE word_labels (
    word TEXT,
    label TEXT,
    FOREIGN KEY(word) REFERENCES words(word),
    UNIQUE(word, label)
);

CREATE INDEX idx_word_labels_label ON word_labels(label);
"""

def save_dict(dt: dict, conn: sqlite3.Connection, cur: sqlite3.Cursor):
    """saves the dictionary at checkpoints to the database. helper to generate_dictionary"""
    cp = deepcopy(dt)
    for word, deets in cp.items():
        # convert label sets to lists
        deets["labels"] = list(deets["labels"])

        # insert items to database
        cur.execute("INSERT OR IGNORE INTO words (word) VALUES (?)", (word,))
        for stress in deets["stress"]:
            cur.execute("INSERT OR IGNORE INTO word_stress (word, stress) VALUES (?, ?)", (word, stress))
        for label in deets["labels"]:
            cur.execute("INSERT OR IGNORE INTO word_labels (word, label) VALUES (?, ?)", (word, label))
    
    conn.commit()

def make_database(out_path: str="corpus/dictionary.db"):
    """generates the database from the above schema. helper to generate_dictionary."""
    conn = sqlite3.connect(out_path)
    cur = conn.cursor()

    # Create tables
    cur.executescript(dict_database_schema)
    return conn, cur