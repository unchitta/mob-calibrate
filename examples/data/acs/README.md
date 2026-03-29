Place raw ACS data extracts/folder in this directory.
Modify file name/path as necessary in examples/01_process_acs.ipynb.

The exact files as used in the 01_process_acs.ipynb example are obtained from data.census.gov using the following steps:

1. Go to data.census.gov
2. Search for the appropriate table names (e.g. B19001 or B01001)
3. Select the appropriate table from the search result
4. Use the following Geography filters:
	1. Select Metropolitan/Micropolitan Statistical Area and choose the desired target area
	2. Select Block Group -> Within Other Geographies -> Metropolitan/Micropolitan Statistical Area -> choose the corresponding desired target area -> All Block Groups within [selected targret area]
5. Back in the data pane, choose the Census ACS year appropriate for your data. 2020: ACS 5-Year Estimates Detailed Tables were used in the examples.
6. Download the data. The resulting files should then be placed in this directory. Note that this generates Census tables in which all but the last row are estimates for Census block groups and the last row contains the estimate for the entire CBSA.