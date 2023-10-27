Each of the following files have two versions, with \_final or \_pilot at the end of their name. The final data refers to the data analyzed in the paper; the pilot data's initial recall is what the CMR parameters were fit to.

#### rawtrialdata.csv
Uneditted trial data output directly from online experiment.

#### cleantrialdata.csv
CSV file broken into columns to more easily read. Run the respective psiturkdatacleaner.py to obtain.

#### formatdata.py; formatdata_cuedtrials.ipynb
Python files for taking the cleantrialdata file to (1) exclude participants based on pre-registered criteria and then (2) covert it to text files to be analyzed. formatdata.py should be run first to achieve output for all participants (included uncued trials). Then, formatdata_cuedtrials.ipynb can be run to take the text files created by formatdata.py and exclude uncued trials. (Note you may need to remove \_pilot or \_final from the end of these files to run as these are simply here to identify which files go with which files).

For pilot data, this will generate the files used to fit CMR to participants' initial recall behaviors. These include (located in the fitCMR/pilotdata folder): 
<pre>
subject_id.txt: Specifies which trials in recs.txt, pres.txt, recs_cats.txt, and pres_cats.txt belong to which subject. In psiturk this just assigns 1 number to each subject.

recs.txt: The recalls of the subjects with each line being a new trial

recs_cats.txt: Similar to recs.txt but it stores the categories of the words in recs.txt with each category having a category number which can be found in wordpool.csv.

pres.txt: The words that were shown in each trial prior to recall.

pres_cats.txt: Similar to pres.txt but it stores the categories of the words in pres.txt with each category having a category number which can be found in wordpool.csv.

GloVe.txt: Stores the GloVe cosine similarity between each word. Indexes determined by location in wordpool.csv. Index i,j is similarity between word i and word j. If you would like to refresh this which you probably won't you could use the '--refresh' or '-r' tag.
</pre>

For final data, this will generate the files used to simulate the experiment in CMR and run analysis on the empirical data. These include all the same files as the pilot data as well as (located in the data/finaldata_opt folder; note though this is not an exhaustive list as some are also generated within final_analyses.ipynb):
<pre>
pres_remaining.txt: the list of words, in order of their presentation, that subjects did not recall initially, and therefore, are available to be recalled during the reminder session

rmdr.txt: index of reminder in remaining word list (-1 appended if no reminder requested)

post_rmdr.txt: words (if any) recalled after reminder for each trial. (-1 if no reminder was shown, to be distinct from recall of 0 after a cue).

post_rmdr_amt.txt: amount of words recalled after reminder for each trial (-1 represents if no reminder was requested, to be distinct from 0 words recalled even when a reminder was provided)

post_rmdr_fromcue.txt: like post_rmdr, but appending to the front of the list is the reminder's wordpool index (for CRP plot). If no cue was displayed, a -1 is appended.

post_rmdr_fromlast.txt: also like post_rmdr and post_rmdr_fromcue where at front of list is the wordpool index of the last word recalled during the initial recall session (also for CRP plot). If no words were initially recalled, -1 is appended.

cue_time.txt: time (in ms) in each recall trial a cue was requested; (90000 = no cue requested because that is the length of the entire recall session)

display_time.txt: how long it took CMR to run in live experiment for each trial. Should be 3000ms if buffer time worked or 0 if no cue shown; otherwise, may be longer.

rmdr_types.txt: if the trial used a CMR-best, CMR-worst, or random reminder. If no reminder shown, 'none' is appended.
</pre>