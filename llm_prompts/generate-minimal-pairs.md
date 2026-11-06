We are making pairs of short English sentences for a language model experiment.

For each item, write a sentence with a blank (___), where a cue word before the blank makes the blank mean "$donor" ($donor_definition). Then give a second cue word: when it replaces the first cue word, the blank must mean "$target" ($target_definition) instead.
$cue_instruction

Rules:
- The reader sees only the words before the blank, so the cue must come before ___. Put ___ late in the sentence, after the cue (for example as the object of a verb), not as the subject at the start: "The ___ slithered away." is wrong, because its cue "slithered" comes after the blank.
- The first cue word must be written in the sentence exactly once, before ___. The second sentence is the same sentence with only the cue word replaced.
- Each cue word is one common word without spaces or hyphens, and the two cue words are different. Choose two cue words of about the same length.
- The cue words are not the words that fill the blank: do not use "$donor" or "$target" anywhere in the sentence or as cue words.
- Both sentences must be natural: the sentence as written, with "$donor" in the blank, and the sentence with the second cue word, with "$target" in the blank.
- Apart from the cue word, the sentence must fit both meanings: no other word may suggest either one (for example, avoid "for dinner" in a sentence that should also work for "car").
- Write the blank as ___ exactly once, as a word of its own (not "___s").
- Keep the sentence under 15 words.

Write $number different items.
