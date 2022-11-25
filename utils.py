def session_to_features(session):
    typing_features = []
    for idx, (tstamp, event, key) in enumerate(session):

        if event == 1:
            continue

        # get the release event
        for idx_rel, (tstamp_rel, event_rel, key_rel) in enumerate(session[idx + 1:]):
            if event_rel == 1 and key_rel == key:

                hl = tstamp_rel - tstamp
                
                # next pressed key
                for idx_next, (tstamp_next, event_next, key_next) in enumerate(session[idx + 1:]):
                    if event_next == 0 and key_next != key:

                        il = tstamp_next - tstamp_rel
                        pl = tstamp_next - tstamp

                        # next release key
                        for idx_next_rel, (tstamp_next_rel, event_next_rel, key_next_rel) in enumerate(session[idx + 1:]):
                            if event_next_rel == 1 and key_next_rel == key_next:
                                rl = tstamp_next_rel - tstamp_rel
                                # print("key: {}, hold: {}, inter: {}, press: {}, release: {}".format(key, hl, il, pl, rl))
                                # transform to s
                                typing_features.append([key, hl / 1000, il / 1000, pl / 1000, rl / 1000])
                                break
                        break
                break
    return typing_features