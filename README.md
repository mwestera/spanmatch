# spanmatch

This tool lets you answer: to what extent did two humans or computer systems highlight the same _character spans_ in a text?

In particular, it lets you figure this out, in case you have _multiple tiers_ of spans that are associated, such as questions and their respective answers.

Indeed, that is the main use-case for which this tool was developed:

1. We have a paragraph containing questions, and a (separate) paragraph containing answers.
2. Neither questions nor answers correspond neatly to individual sentences.
3. Not all questions are necessarily answered. 
4. The spans expressing questions or answers may overlap.
5. Spans can be discontiguous.


## Install

```bash
$ pip install git+https://github.com/mwestera/spanviz
$ pip install git+https://github.com/mwestera/spanmatch
```

This makes available `spanmatch` to import in Python, as well as the `spanmatch` command-line interface.

## Module usage

```python
import spanmatch

annotations_per_document = [
    {
        "id": "document1",
         "text": {
             "questions": "Herinnert u zich mijn schriftelijke vragen over het weigeren van mannelijke artsen door gesluierde vrouwen? Herinnert u zich uw antwoord dat zoveel mogelijk recht moet worden gedaan aan de keuzevrijheid van de cli\u00ebnt, maar dat er wel grenzen zijn? Kunt u aangeven waar deze grenzen liggen en waarop deze zijn gebaseerd? .",
             "answers": "Het zou alleen mogelijk zijn om dat per specifieke situatie precies aan te geven, maar de keuzevrijheid van de pati\u00ebnt mag er niet toe leiden dat de veiligheid van de zorg in gevaar komt. "},
         "spans": {
             "mw": {"questions": [[{"start": 0, "end": 107}], [{"start": 50, "end": 80}]], "answers": [[{"start": 87, "end": 187}], [{"start": 108, "end": 247}], [{"start": 248, "end": 321}]]},
             "js": {"questions": [[{"start": 50, "end": 80}], [{"start": 0, "end": 107}]], "answers": [[{"start": 248, "end": 321}], [{"start": 0, "end": 187}]]}
         }
     },
    # ... more documents
]

aggregator = spanmatch.ComparisonAggregator(['mw', 'js'], ['questions', 'answers'])

for doc in annotations_per_document:
    spanmatch.flatten_spans(doc)    # to turn {'start': 41, 'end': 66} dicts into [41, 66] pairs
    aggregator.process(doc)

scores = aggregator.compute_scores()
print(scores)

```


## Command-line usage (`spanmatch` command)

Input is a `.jsonl` file containing JSON entries, one line for each 'document', that should have three keys: `id`, `text` and `spans`.

- The `id` represents a unique identifier for each document.
- The field `text` should contain the text, or a dictionary mapping tier names to texts (in our case 'questions', 'answers').
- The field `spans` should map annotator (or model) names to their annotated spans. For each annotator, the spans are stored in another mapping from tier names ('questions', 'answers') to lists of spans.

Since a span can be discontiguous, each span is represented as a _list_ of dictionaries with `start` and `end` keys.  

Example input:

```json lines
{"id": "ah-tk-20102011-1016.6", "text": {"questions": "Bent u het eens met het IVO dat betere voorlichting oneigenlijk gebruik kan voorkomen en afkicken bevordert? Zo ja, hoe gaat u dit bevorderen? Zo nee, waarom niet?", "answers": "Gelet op de aard en de omvang van oneigenlijk gebruik van Ritalin zoals nu bekend wil ik mij vooralsnog in deze beperken tot het monitoren en voorlichten met behulp van het bestaande instrumentarium. "}, "spans": {"annotator1": {"questions": [[{"start": 0, "end": 108, "score": 5, "text": "Bent u het eens met het IVO dat betere voorlichting oneigenlijk gebruik kan voorkomen en afkicken bevordert?"}], [{"start": 109, "end": 142, "score": 5, "text": "Zo ja, hoe gaat u dit bevorderen?"}], [{"start": 143, "end": 163, "score": 5, "text": "Zo nee, waarom niet?"}]], "answers": [[], [], [{"start": 0, "end": 81, "score": 5, "text": "Gelet op de aard en de omvang van oneigenlijk gebruik van Ritalin zoals nu bekend"}]]}, "annotator2": {"questions": [[{"start": 0, "end": 108, "score": 5, "text": "Bent u het eens met het IVO dat betere voorlichting oneigenlijk gebruik kan voorkomen en afkicken bevordert?"}], [{"start": 109, "end": 142, "score": 5, "text": "Zo ja, hoe gaat u dit bevorderen?"}], [{"start": 143, "end": 163, "score": 5, "text": "Zo nee, waarom niet?"}]], "answers": [[{"start": 112, "end": 199, "score": 5, "text": "beperken tot het monitoren en voorlichten met behulp van het bestaande instrumentarium."}], [], [{"start": 0, "end": 65, "score": 5, "text": "Gelet op de aard en de omvang van oneigenlijk gebruik van Ritalin"}]]}}}
{"id": "ah-tk-20092010-3299.6", "text": {"questions": "Hoe beoordeelt u het nut van dagbesteding voor (ex-)psychiatrisch pati\u00ebnten? Wat is uw reactie op de verklaring van cli\u00ebnt W.: \u00abAls ik steeds alleen thuis zou zitten, ben ik bang dat het helemaal verkeerd gaat. Ik ben heel ziek geweest\u00bb?", "answers": "Het gaat hier niet om het nut van de geboden zorg, maar om de redelijkheid daarvoor een eigen bijdrage te heffen. Bij de bijdrage wordt rekening gehouden met de draagkracht van betrokkene. "}, "spans": {"annotator1": {"questions": [[{"start": 0, "end": 76, "score": 5, "text": "Hoe beoordeelt u het nut van dagbesteding voor (ex-)psychiatrisch pati\u00ebnten?"}], [{"start": 77, "end": 237, "score": 5, "text": "Wat is uw reactie op de verklaring van cli\u00ebnt W.: \u00abAls ik steeds alleen thuis zou zitten, ben ik bang dat het helemaal verkeerd gaat. Ik ben heel ziek geweest\u00bb?"}]], "answers": [[{"start": 0, "end": 50, "score": 2, "text": "Het gaat hier niet om het nut van de geboden zorg,"}], []]}, "annotator2": {"questions": [[{"start": 0, "end": 76, "score": 5, "text": "Hoe beoordeelt u het nut van dagbesteding voor (ex-)psychiatrisch pati\u00ebnten?"}], [{"start": 77, "end": 237, "score": 5, "text": "Wat is uw reactie op de verklaring van cli\u00ebnt W.: \u00abAls ik steeds alleen thuis zou zitten, ben ik bang dat het helemaal verkeerd gaat. Ik ben heel ziek geweest\u00bb?"}]], "answers": [[{"start": 0, "end": 50, "score": 1, "text": "Het gaat hier niet om het nut van de geboden zorg,"}], [{"start": 114, "end": 188, "score": 5, "text": "Bij de bijdrage wordt rekening gehouden met de draagkracht van betrokkene."}]]}}}
```

Assuming we have such data in `test.jsonl`, we can do:

```bash
$ cat test.jsonl | spanmatch
```