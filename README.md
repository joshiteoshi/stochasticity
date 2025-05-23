# stochasticity

This is the repository for *stochasticity*, a computational poetry project.

This code was written to solve a very stupid problem: given an English word stress sequence, can we segment it, label the segments, and randomly fill the words to create syntactically correct nonsense poetry?
If that seems like an exercise in futility, you are 100% correct.
Here are some excerpts from *stochasticity*:

> between untold compute or ring<br/>
> aloft a rate between<br/>
> the lightning than the none was fear<br/>
> and said him out this jaw
>
> and waters in that price with void<br/>
> beyond enough event<br/>
> behind a weight beside are found<br/>
> a lyrics live with wit
>
> or as a wings together die<br/>
> that anger like a boot<br/>
> with which good match or a good do<br/>
> a feel in meet to this

Still interested in running this for some reason?
Follow the instructions below.

## How to get this mess running

This project was done in Python 3.12.4.
I recommend using `pyenv` to set up a virtual environment to run and install everything.
From whatever environment you prefer, do:

```
pip install requirements.txt
```

Which will install all the project's dependencies.

### Training

Use any corpus of poetry you like.
Originally, I used the gutenberg poetry corpus provided by [Parrish](https://github.com/aparrish/gutenberg-poetry-corpus).
However, since it contained lots of obscure words, I opted to instead use my own poetry as the training set.

The dataset generator functions expect a list of strings.
Fiddle with the training notebook to get a model for the wrapper class.

### Just making it run

I haven't put up the models yet. I will get to that eventually.

## Model & Logic

*stochasticity* generates poetry from a stress sequence by splitting the stress sequence into words, generating POS tags for each word, and finally randomly querying a dictionary for words satisfying the stress pattern and POS tag.
Since words can have multiple stress sequences and POS tags, the database is generated by figuring getting each stress sequence from the CMU pronouncing dictionary, and using another POS labeller to find every context a word can occur in.
Decoding a sequence is done using the following pipeline:

 1. Stress splitting

 Stress sequences are split by sampling a geometric distribution to determine the next word's syllable count.
 By default, this is parametised to `p=0.8`, though it can be lowered to generate more long words.
 If too high, the probability of an unsatisfiable sequence increases.

 2. Stress-sequence labelling

 A simple encoder + GRU model is used to generate likely sequences of labels.
 The code for the model is found in `stochasticity.py`.
 Since spaCy is used to generate the POS tags, we use the same list of tags for sequence labelling.

 3. Word decoder

 Using the previously generated splits and POS labels, an sqlite3 database is queried to find all words satisfying both.
 This process is fallible, and the wrapper class in `generator.py` retries with new predicted labels some number of times before declaring the sequence unsatisfiable.
 If the line succeeds, we get our new nonsense line.

The wrapper in `generator.py` can take in a stanza of stress sequences and returns a stanza of nonsense poetry.
