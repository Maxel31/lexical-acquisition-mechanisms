We are making pairs of short English sentences for a language model experiment.

In each pair, the two sentences are the same except for one cue word, which comes before a blank (___).
- In the first sentence, the cue makes the blank mean "$donor" ($donor_definition).
- In the second sentence, the cue makes the blank mean "$target" ($target_definition).
$cue_instruction

Rules:
- Each sentence must be natural when the blank is filled with the word it is meant for.
- Write the blank as ___ exactly once in each sentence, after the cue word.
- The cue word must be one word without spaces or hyphens, and the two cue words must be different.
- Do not use the words "$donor" or "$target" anywhere in the sentences.
- Keep each sentence under 15 words.

Write $number different pairs.
