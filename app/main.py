from __future__ import print_function, division
import getopt
import math
import os
import sys
import flask
import flask_compress
import geojson
import json
import pymongo
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import time

from collections import Counter
from sklearn.linear_model import RidgeCV
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from vincenty import vincenty

#   Initialize globals

app = flask.Flask(__name__)

#   Configure Flask to not sort or prettify JSON because we're sending 
#   lots of data

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['JSON_SORT_KEYS'] = False

#   Compress Flask responses with gzip

flask_compress.Compress(app)

#   Globals

mongo_collection = None
canon = None
topFeatures = None

#   Helper function to report memory usage

def report_memory_usage():
    result = {'peak' : 0, 'rss' : 0}
    with open('/proc/{}/status'.format(os.getpid())) as status:
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1]) // 1024
    print(', '.join(['{}: {}MB'.format(key, value) for key, value in result.items()]))

#   Decorator for caching results

result_cache = {}
def memoize(func):
    global result_cache
    def func_wrapper(*args, **kwargs):
        key = tuple(' '.join([str(arg) for arg in args]))
        if key in result_cache:
            result = result_cache[key]
        else:
            result = func(*args, **kwargs)
            result_cache[key] = result
        return result

    return func_wrapper
    

#   Database query

@memoize
def find(criteria, fields, limit=0):
    fields = { field: 1 for field in fields }
    if not '_id' in fields:
        fields['_id'] = 0

    radius = 0.0
    result = list(mongo_collection.find(criteria, fields).limit(limit))
    if 'loc' in criteria:
        center = criteria['loc']['$nearSphere']['$geometry']['coordinates']
        radius = 1609.34 * vincenty((center[1], center[0]),  # convert miles to meters
            (result[-1]['latitude'], result[-1]['longitude']))

    if fields['_id']:
        for r in result:
            r['_id'] = str(r['_id'])

    print('\ncriteria = {}\nfields = {}\n{} results ({:.2f} meters)\n'.format(criteria, 
        fields, len(result), radius))
    return sorted(result, key=lambda x: x['price']), radius

#   Predictor

def get_features_matrix(list_or_iter):
    listings = []
    for features in list_or_iter:
        listings.append({})
        for f in features.lower().replace('-', ' ').split('\n'):
            f = canon.get(f, f)
            if f in topFeatures:
                f_name = 'f_' + f.replace(' ', '_')
                listings[-1][f_name] = 1    
    return pd.DataFrame.from_dict(listings).fillna(0)


def predict(obs, listings):
    
    #   Put listings in DataFrame

    columns = ['bedrooms', 'bathrooms', 'features', 'price']
    df = pd.DataFrame([[listing[col] for col in columns] for listing in listings], columns=columns).dropna()
    df = df.join(get_features_matrix(df.features)).drop('features', axis=1)
    print('{} rows, {} columns'.format(df.shape[0], df.shape[1]))

    #   Fit model

    X = df.drop('price', axis=1)
    y = df['price']
    model = RidgeCV(cv=3)
    model.fit(X, np.log10(y))
    resid = np.power(10, model.predict(X)) - y

    #   Predict observation

    X.loc[0] = np.zeros(X.shape[1])
    X.loc[0, 'bedrooms'] = obs['beds'][0] if 'beds' in obs else 1.0
    X.loc[0, 'bathrooms'] = obs['bath'][0] if 'bath' in obs else 1.0
    for section in ['unit', 'bldg']:
        for feature in obs.get(section, []):
            col = 'f_' + feature.replace(' ', '_')
            if col in X.columns:
                X.loc[0, col] = 1.0

    for col in X:
        if X.loc[0, col] != 0:
            print('{} : {}'.format(col, X.loc[0, col]))

    y_pred = np.power(10, model.predict(X[:1]))[0]
    print("predicted: {}".format(y_pred))

    #   Compose result

    result = { 
        'predict' : int(y_pred), 
        'std' : int(np.std(resid))
    }

    #   Clean up and return result

    del df, X, y
    return result

#   Flask: data

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def send_js(path):
    print('sending {}'.format(path))
    return flask.send_from_directory('.', path)

#   Flask: actions

@app.route('/submit', methods=['POST'])
def submit():
    #	Parse request

    data = flask.request.json

    criteria = {
        'price' : { '$lt' : 1e5 }
    }

    if 'latitude' in data and 'longitude' in data:
        criteria['loc'] = { 
            '$nearSphere' : {
                '$geometry' : geojson.Point((float(data['longitude']), float(data['latitude']))), 
                '$maxDistance' : data.get('distance', 500) 
                }
            }

    if 'beds' in data:
        l1 = filter(lambda x: x != 4, data['beds'])
        l2 = filter(lambda x: x == 4, data['beds'])
        q1 = { '$in' : l1 } if 1 < len(l1) else l1[0] if 1 == len(l1) else None
        q2 = { '$gte' : l2[0] } if l2 else None
        if q1 and q2:
            criteria['$or'] = [{'bedrooms' : q1}, {'bedrooms' : q2}]
        else:
            criteria['bedrooms'] = q2 if q2 else q1

    if 'bath' in data:
        l1 = filter(lambda x: x != 3, data['bath'])
        l2 = filter(lambda x: x == 3, data['bath'])
        q1 = { '$in' : l1 } if 1 < len(l1) else l1[0] if 1 == len(l1) else None
        q2 = { '$gte' : l2[0] } if l2 else None
        if q1 and q2:
            criteria['$or'] = [{'bathrooms' : q1}, {'bathrooms' : q2}]
        else:
            criteria['bathrooms'] = q2 if q2 else q1

    #   Get listings

    columns = ['loc', 'latitude', 'longitude', 'price', 'bedrooms', 'bathrooms', 'features', 'url']
    listings, radius = find(criteria, columns, data.get('limit', 0))

    #   Get Probability Distribution

    prices = [l['price'] for l in listings]
    if prices:
        pdf = stats.norm.pdf(prices, np.mean(prices), np.std(prices))
        if not np.isnan(pdf).any():
            for i, l in enumerate(listings):
                l['pdf'] = pdf[i]
        del pdf


    #   Predict price

    if 'loc' in criteria:
        prediction = predict(data, listings)
        print(prediction)
    else:
        prediction = None

    #	Package and return result
    
    start = time.time()
    result = flask.jsonify(prediction=prediction, listings=listings, radius=radius)
    print("elapsed time:", time.time() - start)

    del listings, prices
    report_memory_usage()
    return result

#   Initialize application

def app_init():
    global canon
    global topFeatures

    #   Load listings

    global mongo_collection
    mongo_client = pymongo.MongoClient('ec2-34-198-246-43.compute-1.amazonaws.com', 27017)
    mongo_collection = mongo_client.renthop2.listings
    print('{} listings'.format(mongo_collection.count()))

    #   Load features

    with open('synonyms.json') as fd:
        synomyns = json.load(fd)
    canon = {alias : term for s in synomyns for term, aliases in s.items() for alias in aliases}

    all_features = Counter()
    df = pd.DataFrame(list(mongo_collection.find({}, ['features']))).dropna()
    for l in df.features:
        unit_features = l.lower().replace('-', ' ').split('\n')
        all_features.update([canon.get(f, f) for f in unit_features if f])
    del all_features['-']

    topFeatures = dict(filter(lambda x: x[1] > 100, all_features.most_common()))
    print('{} features'.format(len(all_features)))
    del synomyns, df, all_features

#   Main

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'dh:', ['debug', 'host='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    debug = False
    host = '0.0.0.0'

    for opt, arg in opts:
        if opt in ['-d', '--debug']:
            debug = True
        elif opt in ['-h', '--host']:
            host = arg

    app_init()

    if debug:
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 60;
    app.run(host=host, debug=debug)


if __name__ == '__main__':
    main(sys.argv[1:])
