
def session_to_features_typenet(session):
    """
    Transform a sesssion into the features used for training.
    - from TypeNet Paper

    Trasnsform timestamp + event into a feature vector containing:
    - Hold latency
    - Inter-key latency
    - Press latency
    - Release latency

    Hot take:

    !!! It's equivalent to using bigrams of events. !!!
    Question:
    Why use bigrams when the paper uses transformers?

    """
    typing_features = []
    for idx, (tstamp, event, key) in enumerate(session):

        if event == 0:
            continue

        # get the release event
        for idx_rel, (tstamp_rel, event_rel, key_rel) in enumerate(session[idx + 1:]):
            if event_rel == 0 and key_rel == key:

                hl = tstamp_rel - tstamp
                
                # next pressed key
                for idx_next, (tstamp_next, event_next, key_next) in enumerate(session[idx + 1:]):
                    if event_next == 1 and key_next != key:

                        il = tstamp_next - tstamp_rel
                        pl = tstamp_next - tstamp

                        # next release key
                        for idx_next_rel, (tstamp_next_rel, event_next_rel, key_next_rel) in enumerate(session[idx + 1:]):
                            if event_next_rel == 0 and key_next_rel == key_next:
                                rl = tstamp_next_rel - tstamp_rel
                                # print("key: {}, hold: {}, inter: {}, press: {}, release: {}".format(key, hl, il, pl, rl))
                                # transform to s
                                # normalize keycodes javascript
                                typing_features.append([key/255, hl / 1000, il / 1000, pl / 1000, rl / 1000])
                                break
                        break
                break
    return typing_features

def session_to_features_our(session):
    """
    Our approach to transform a session into features.
    
    """
    pass


def split_dataset():
    """
    Split the dataset into train and test set.
    """
    pass