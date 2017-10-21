Installation Instructions:

To run the app bitcoin_correlations_app.py locally, you will need to have the following items installed:

- Python (3.5+) and the most recent versions of the following modules:
    - Pandas (https://pandas.pydata.org/pandas-docs/stable/install.html)
    - Pandas DataReader (https://pypi.python.org/pypi/pandas-datareader)
    - Numpy  (https://docs.scipy.org/doc/numpy-1.13.0/user/install.html)
    - Bokeh  (https://bokeh.pydata.org/en/latest/docs/installation.html)

Since these instructions are highly OS dependent, I suggest visiting the documentation. The app was created on Linux (Ubuntu 16.04)
For linux, it is quite easy to install Anaconda, which then allows you to download each module by typing:
-    conda install pandas
-    conda install numpy
-    conda install bokeh
-    conda install pandas-datareader

1. Once these modules are installed, open up a command prompt (terminal) and navigate to the directory where bitcoin_correlations_app.py is installed.
2. Next type command: bokeh serve bitcoin_correlations_app.py
3. Your terminal should show something like this:

    justin@JBot:~/PycharmProjects/bitcoin_arbitrage/lab4$ bokeh serve bitcoin_correlations_app.py
    2017-10-15 13:14:39,500 Starting Bokeh server version 0.12.5
    2017-10-15 13:14:39,503 Starting Bokeh server on port 5006 with applications at paths ['/bitcoin_correlations_app']
    2017-10-15 13:14:39,503 Starting Bokeh server with process id: 15725


4. Note the port [5006 in this example] and open up a web browser:

    http://localhost:5006/

5. You should now have access to the Interactive Correlation Visualization.