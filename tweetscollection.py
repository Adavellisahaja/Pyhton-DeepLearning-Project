# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json

# Go to http://apps.twitter.com and create an app.
# The consumer key and secret will be generated for you after
consumer_key = '69SNKxw0qbGY6PPZBcRVLZVpP'
consumer_secret = 'sIUppqVb3mDXDdbnXGst50fL2DvmtcyjakxbJhA7D5vQpt3PNr'

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
access_token = '1039586617445572608-3R3OHxxrH5JY8e9IITG3w3pwkZpGFc'
access_token_secret = 'KrqdNUh5tddGq1SW1SUIeQMBelQcyLzQVOaOzAPfNZG84'


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """

    def on_data(self, data):
        try:
            with open('data1.json', 'a') as outfile:
                json.dump(data, outfile)
            with open('data2.json', 'a') as outputj:
                outputj.write(data)
            with open('tweets.txt', 'a') as tweets:
                tweets.write(data)
                tweets.write('\n')
            outfile.close()
            tweets.close()
            outputj.close()
        except BaseException as e:
            print('problem collecting tweet', str(e))
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)

    stream.filter(track=['stockmarket', 'bitcoin', 'money', 'trading', 'forextrader', 'investment', 'wallstreet', 'stocks', 'entrepreneur',
     'forex', 'trader', 'investor', 'investing', 'cryptocurrency', 'invest', 'business' , 'daytrader', 'binaryoptions', 'forexsignals',
    'profit' ,'success' ,'finance', 'wealth', 'makemoneyonline' ,'forexlifestyle' ,'forextrading' ,'motivation' ,'millionaire', 'entrepreneurship' ,'daytrading'])

