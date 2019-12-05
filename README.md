# tgnews
Data Clustering Contest: Round 1 

**The Task**



- Isolate articles in English and Russian. Your algorithm must sort articles by language, filtering English and Russian articles. Articles in other languages are not relevant for this stage of the contest and may be discarded.

- Isolate news articles. Your algorithm must discard everything except for news articles.

- Group news articles by category. Your algorithm must place news articles into the following 7 categories:

  Society (includes Politics, Elections, Legislation, Incidents, Crime)
  Economy (includes Markets, Finance, Business)
  Technology (includes Gadgets, Auto, Apps, Internet services)
  Sports (includes E-Sports)
  Entertainment (includes Movies, Music, Games, Books, Arts)
  Science (includes Health, Biology, Physics, Genetics)
  Other (news articles that don’t fall into any of the above categories)
- Group similar news into threads. Your algorithm must identify news articles about the same event and group them together into threads, selecting a relevant title for each thread. News articles inside each thread must be sorted according to their relevance (most relevant at the top).

- Sort threads by their relative importance. Your algorithm must sort news threads in each of the categories based on perceived importance (important at the top). In addition, the algorithm must build a global list of threads, indepedent of category, sorted by perceived importance (important at the top).

***pipeline5.py***
- 1 Get files from the source dir and create dataframe
- 2 Detect english and russian files, using cld2 lib
- 3 Perform preliminary clasterisation, using LDA method. Prepare labeled dataset for classification
- 4 Use SGD Classifirer for traing and test data. Predict category for every document
- 5 Use LDA again to detect threads and rank them for every category. Define category - "others" and define files with nesws.
- 6 Save all data in two csv files - stage41.csv (eng) and stage42.csv (ru)





    
