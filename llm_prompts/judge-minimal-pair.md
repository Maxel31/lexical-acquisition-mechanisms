We are checking pairs of short English sentences for a language model experiment. The two sentences differ in one cue word, and the cue word should decide which of two words fits the blank.

Word A: "$donor" ($donor_definition)
Word B: "$target" ($target_definition)
$role_instruction

Sentence 1 with word A: $donor_sentence_with_donor
Sentence 1 with word B: $donor_sentence_with_target
Sentence 2 with word B: $target_sentence_with_target
Sentence 2 with word A: $target_sentence_with_donor

First, in "reason", say briefly how a careful reader would understand each of the four sentences, with each word in the meaning given above. Then answer:
- "sentence_1_fits": true only if "Sentence 1 with word A" is a natural English sentence, and "Sentence 1 with word B" is clearly strange or much less plausible.
- "sentence_2_fits": true only if "Sentence 2 with word B" is a natural English sentence, and "Sentence 2 with word A" is clearly strange or much less plausible.

Answer false when:
- the cue word fits both words about equally (for example, "She spotted the ___.", "They bought the ___." or "He placed the ___." fit almost any thing)
- the sentence is ungrammatical, makes no sense, or is natural only as a joke or a metaphor
- a word is used in a role other than the one given above (for example, a noun where an adjective is needed)
