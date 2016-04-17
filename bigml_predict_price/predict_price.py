def predict_price(address=None,
                  bedrooms=None,
                  full_bathrooms=None,
                  half_bathrooms=None,
                  _type=None,
                  lot_size=None):
    """ Predictor for price
    Code from BigML

        Predictive model by BigML - Machine Learning Made Easy
    """

    import re

    tm_tokens = 'tokens_only'
    tm_full_term = 'full_terms_only'
    tm_all = 'all'

    def term_matches(text, field_name, term):
        """ Counts the number of occurences of term and its variants in text

        """
        forms_list = term_forms[field_name].get(term, [term])
        options = term_analysis[field_name]
        token_mode = options.get('token_mode', tm_tokens)
        case_sensitive = options.get('case_sensitive', False)
        first_term = forms_list[0]
        if token_mode == tm_full_term:
            return full_term_match(text, first_term, case_sensitive)
        else:
            # In token_mode='all' we will match full terms using equals and
            # tokens using contains
            if token_mode == tm_all and len(forms_list) == 1:
                pattern = re.compile(r'^.+\b.+$', re.U)
                if re.match(pattern, first_term):
                    return full_term_match(text, first_term, case_sensitive)
            return term_matches_tokens(text, forms_list, case_sensitive)


    def full_term_match(text, full_term, case_sensitive):
        """Counts the match for full terms according to the case_sensitive
              option

        """
        if not case_sensitive:
            text = text.lower()
            full_term = full_term.lower()
        return 1 if text == full_term else 0

    def get_tokens_flags(case_sensitive):
        """Returns flags for regular expression matching depending on text
              analysis options

        """
        flags = re.U
        if not case_sensitive:
            flags = (re.I | flags)
        return flags


    def term_matches_tokens(text, forms_list, case_sensitive):
        """ Counts the number of occurrences of the words in forms_list in
               the text

        """
        flags = get_tokens_flags(case_sensitive)
        expression = r'(\b|_)%s(\b|_)' % '(\\b|_)|(\\b|_)'.join(forms_list)
        pattern = re.compile(expression, flags=flags)
        matches = re.findall(pattern, text)
        return len(matches)


    term_analysis = {
        "address": {
            "token_mode": 'all',
            "case_sensitive": False,
        },
    }
    term_forms = {
        "address": {
        },
    }
    if (full_bathrooms is None):
        return 235138.42196
    if (full_bathrooms == '2.0'):
        if (half_bathrooms is None):
#### Check ML
            return 182748.6719
        if (half_bathrooms == '0.0'):
            if (_type is None):
                return 150775.85855
            if (_type == 'Single Family Home'):
                if (lot_size is None):
                    return 176811.69525
                if (lot_size == '207781.2'):
                    return 898900
                if (lot_size != '207781.2'):
                    if (lot_size == '49222.8'):
                        return 575000
                    if (lot_size != '49222.8'):
                        if (address is None):
                            return 175090.84025
                        if (term_matches(address, "address", "las") > 0):
                            if (lot_size == '31363.2'):
                                return 529000
                            if (lot_size != '31363.2'):
                                if (lot_size == '23958.0'):
                                    return 479900
                                if (lot_size != '23958.0'):
                                    if (lot_size == '9583.0'):
                                        return 344633.33333
                                    if (lot_size != '9583.0'):
                                        if (lot_size == '90604.8'):
                                            return 450000
                                        if (lot_size != '90604.8'):
                                            if (lot_size == '165528.0'):
                                                return 449000
                                            if (lot_size != '165528.0'):
                                                if (lot_size == '20473.2'):
                                                    return 302099.75
                                                if (lot_size != '20473.2'):
                                                    if (lot_size == '95396.4'):
                                                        return 425000
                                                    if (lot_size != '95396.4'):
                                                        if (lot_size == '42688.8'):
                                                            return 389900
                                                        if (lot_size != '42688.8'):
                                                            if (lot_size == '21344.4'):
                                                                return 255724.83333
                                                            if (lot_size != '21344.4'):
                                                                if (term_matches(address, "address", "89121") > 0):
                                                                    if (lot_size == '20908.8'):
                                                                        return 329000
                                                                    if (lot_size != '20908.8'):
                                                                        return 132923.28947
                                                                if (term_matches(address, "address", "89121") <= 0):
                                                                    if (lot_size == '8276.0'):
                                                                        return 227154.93333
                                                                    if (lot_size != '8276.0'):
                                                                        if (term_matches(address, "address", "89108") > 0):
                                                                            return 142850.20755
                                                                        if (term_matches(address, "address", "89108") <= 0):
                                                                            if (term_matches(address, "address", "ct") > 0):
                                                                                return 197605.52941
                                                                            if (term_matches(address, "address", "ct") <= 0):
                                                                                return 167808.42521
                        if (term_matches(address, "address", "las") <= 0):
                            if (bedrooms is None):
                                return 255360.34783
                            if (bedrooms == '4.0'):
                                return 172200
                            if (bedrooms != '4.0'):
                                return 272867.78947
            if (_type != 'Single Family Home'):
                if (bedrooms is None):
                    return 125362.70254
                if (bedrooms == '1.0'):
                    if (address is None):
                        return 330223.77778
                    if (term_matches(address, "address", "dr") > 0):
                        return 620722
                    if (term_matches(address, "address", "dr") <= 0):
                        if (term_matches(address, "address", "st") > 0):
                            return 59950
                        if (term_matches(address, "address", "st") <= 0):
                            return 278436.66667
                if (bedrooms != '1.0'):
                    if (lot_size is None):
                        return 119698.34101
                    if (lot_size == '2308.0'):
                        return 685000
                    if (lot_size != '2308.0'):
                        if (_type == 'Single Family Home; Ready to build'):
                            return 342160
                        if (_type != 'Single Family Home; Ready to build'):
                            if (lot_size == '3049.0'):
                                return 163453.06897
                            if (lot_size != '3049.0'):
                                if (lot_size == '7579.0'):
                                    return 304000
                                if (lot_size != '7579.0'):
                                    if (_type == 'Mfd/Mobile Home'):
                                        return 68182.41333
                                    if (_type != 'Mfd/Mobile Home'):
                                        if (address is None):
                                            return 102720.12207
                                        if (term_matches(address, "address", "89123") > 0):
                                            return 164842.85714
                                        if (term_matches(address, "address", "89123") <= 0):
                                            return 100609.15534
        if (half_bathrooms != '0.0'):
            if (lot_size is None):
                return 226323.99072
            if (lot_size == '7840.0'):
                return 360254.08696
            if (lot_size != '7840.0'):
                if (_type is None):
                    return 216522.60263
                if (_type == 'Condo/Townhome/Row Home/Co-Op'):
                    if (lot_size == '2700.0'):
                        return 596626.66667
                    if (lot_size != '2700.0'):
                        if (lot_size == '2482.0'):
                            return 469900
                        if (lot_size != '2482.0'):
                            if (address is None):
                                return 130026.74227
                            if (term_matches(address, "address", "las") > 0):
                                if (lot_size == '2613.0'):
                                    return 201240
                                if (lot_size != '2613.0'):
                                    return 114890.71605
                            if (term_matches(address, "address", "las") <= 0):
                                return 210862.88889
                if (_type != 'Condo/Townhome/Row Home/Co-Op'):
                    if (lot_size == '16552.8'):
                        return 779900
                    if (lot_size != '16552.8'):
                        if (lot_size == '2178.0'):
                            return 154402.29412
                        if (lot_size != '2178.0'):
                            if (lot_size == '1742.0'):
                                return 147901.75676
                            if (lot_size != '1742.0'):
                                if (bedrooms is None):
                                    return 233360.52011
                                if (bedrooms == '2.0'):
                                    return 322229.27586
                                if (bedrooms != '2.0'):
                                    if (lot_size == '27007.2'):
                                        return 697000
                                    if (lot_size != '27007.2'):
                                        if (lot_size == '55756.8'):
                                            return 679900
                                        if (lot_size != '55756.8'):
                                            if (lot_size == '174240.0'):
                                                return 444930
                                            if (lot_size != '174240.0'):
                                                if (lot_size == '8712.0'):
                                                    return 329511.23077
                                                if (lot_size != '8712.0'):
                                                    if (lot_size == '2613.0'):
                                                        if (address is None):
                                                            return 181536.19298
                                                        if (term_matches(address, "address", "unit") > 0):
                                                            return 275360
                                                        if (term_matches(address, "address", "unit") <= 0):
                                                            return 176323.75926
                                                    if (lot_size != '2613.0'):
                                                        if (lot_size == '3484.0'):
                                                            return 194032.17391
                                                        if (lot_size != '3484.0'):
                                                            if (lot_size == '49658.4'):
                                                                return 525000
                                                            if (lot_size != '49658.4'):
                                                                if (lot_size == '3049.0'):
                                                                    return 195906.86957
                                                                if (lot_size != '3049.0'):
                                                                    if (lot_size == '23086.8'):
                                                                        return 499900
                                                                    if (lot_size != '23086.8'):
                                                                        if (lot_size == '45302.4'):
                                                                            return 499900
                                                                        if (lot_size != '45302.4'):
                                                                            if (lot_size == '23958.0'):
                                                                                return 489900
                                                                            if (lot_size != '23958.0'):
                                                                                if (lot_size == '19166.4'):
                                                                                    return 489900
                                                                                if (lot_size != '19166.4'):
                                                                                    if (lot_size == '10018.8'):
                                                                                        return 307679
                                                                                    if (lot_size != '10018.8'):
                                                                                        if (_type == 'Mfd/Mobile Home'):
                                                                                            return 83900
                                                                                        if (_type != 'Mfd/Mobile Home'):
                                                                                            if (lot_size == '3920.0'):
                                                                                                return 210642.1746
                                                                                            if (lot_size != '3920.0'):
                                                                                                if (lot_size == '4791.0'):
                                                                                                    return 214012.96721
                                                                                                if (lot_size != '4791.0'):
                                                                                                    if (lot_size == '4356.0'):
                                                                                                        return 217522.76
                                                                                                    if (lot_size != '4356.0'):
                                                                                                        if (lot_size == '5227.0'):
                                                                                                            return 225333.23404
                                                                                                        if (lot_size != '5227.0'):
                                                                                                            if (lot_size == '13939.2'):
                                                                                                                return 439888
                                                                                                            if (lot_size != '13939.2'):
                                                                                                                return 257132.63256
    if (full_bathrooms != '2.0'):
        if (full_bathrooms == '1.0'):
            if (half_bathrooms is None):
                return 115873.4037
            if (half_bathrooms == '0.0'):
                if (lot_size is None):
                    return 100071.01031
                if (lot_size == '93218.4'):
                    return 399000
                if (lot_size != '93218.4'):
                    if (lot_size == '197762.4'):
                        return 325000
                    if (lot_size != '197762.4'):
                        if (lot_size == '46609.2'):
                            return 320000
                        if (lot_size != '46609.2'):
                            if (_type is None):
                                return 73710.71622
                            if (_type == 'Single Family Home'):
                                return 102023.83333
                            if (_type != 'Single Family Home'):
                                return 54406.31818
            if (half_bathrooms != '0.0'):
                if (bedrooms is None):
                    return 156211.09211
                if (bedrooms == '1.0'):
                    if (address is None):
                        return 261272.1
                    if (term_matches(address, "address", "89121") > 0):
                        return 56300
                    if (term_matches(address, "address", "89121") <= 0):
                        return 302266.52
                if (bedrooms != '1.0'):
                    if (lot_size is None):
                        return 87693.04348
                    if (lot_size == '10018.8'):
                        return 349000
                    if (lot_size != '10018.8'):
                        if (lot_size == '18730.8'):
                            return 250000
                        if (lot_size != '18730.8'):
                            return 78586.66667
        if (full_bathrooms != '1.0'):
            if (half_bathrooms is None):
##### CHECK own
                return 372450.29035
            if (half_bathrooms == '0.0'):
                if (full_bathrooms == '3.0'):
                    if (address is None):
                        return 290111.88661
                    if (term_matches(address, "address", "unit") > 0):
                        if (term_matches(address, "address", "rd") > 0):
                            return 954333
                        if (term_matches(address, "address", "rd") <= 0):
                            return 340241.5
                    if (term_matches(address, "address", "unit") <= 0):
                        if (term_matches(address, "address", "las") > 0):
                            if (lot_size is None):
                                return 282762.73813
                            if (lot_size == '16552.8'):
                                return 657500
                            if (lot_size != '16552.8'):
                                if (lot_size == '3920.0'):
                                    return 202011.30303
                                if (lot_size != '3920.0'):
                                    if (lot_size == '3484.0'):
                                        return 193250
                                    if (lot_size != '3484.0'):
                                        if (term_matches(address, "address", "89121") > 0):
                                            return 185731.76471
                                        if (term_matches(address, "address", "89121") <= 0):
                                            if (term_matches(address, "address", "89122") > 0):
                                                return 177389.83333
                                            if (term_matches(address, "address", "89122") <= 0):
                                                if (lot_size == '30492.0'):
                                                    return 642900
                                                if (lot_size != '30492.0'):
                                                    if (term_matches(address, "address", "89108") > 0):
                                                        return 202257.69231
                                                    if (term_matches(address, "address", "89108") <= 0):
                                                        if (lot_size == '2613.0'):
                                                            return 174412.85714
                                                        if (lot_size != '2613.0'):
                                                            if (lot_size == '4356.0'):
                                                                return 246885.97368
                                                            if (lot_size != '4356.0'):
                                                                if (lot_size == '3049.0'):
                                                                    return 206726.16667
                                                                if (lot_size != '3049.0'):
                                                                    if (lot_size == '4791.0'):
                                                                        return 261781.97826
                                                                    if (lot_size != '4791.0'):
                                                                        if (term_matches(address, "address", "ct") > 0):
                                                                            if (lot_size == '10454.4'):
                                                                                return 639000
                                                                            if (lot_size != '10454.4'):
                                                                                if (lot_size == '8276.0'):
                                                                                    return 463679.8
                                                                                if (lot_size != '8276.0'):
                                                                                    if (lot_size == '19602.0'):
                                                                                        return 575000
                                                                                    if (lot_size != '19602.0'):
                                                                                        return 328959.48148
                                                                        if (term_matches(address, "address", "ct") <= 0):
                                                                            if (lot_size == '130680.0'):
                                                                                return 369377.05882
                                                                            if (lot_size != '130680.0'):
                                                                                if (lot_size == '31798.8'):
                                                                                    return 570000
                                                                                if (lot_size != '31798.8'):
                                                                                    if (_type is None):
                                                                                        return 300444.90549
                                                                                    if (_type == 'Multi-Family Home; Ready to build'):
                                                                                        return 168913.5
                                                                                    if (_type != 'Multi-Family Home; Ready to build'):
                                                                                        if (lot_size == '20037.6'):
                                                                                            return 445866.66667
                                                                                        if (lot_size != '20037.6'):
                                                                                            if (lot_size == '15246.0'):
                                                                                                return 550000
                                                                                            if (lot_size != '15246.0'):
                                                                                                if (lot_size == '22215.6'):
                                                                                                    return 421373.75
                                                                                                if (lot_size != '22215.6'):
                                                                                                    if (bedrooms is None):
                                                                                                        return 298408.79747
                                                                                                    if (bedrooms == '7.0'):
                                                                                                        return 132500
                                                                                                    if (bedrooms != '7.0'):
                                                                                                        if (lot_size == '12632.4'):
                                                                                                            return 412475
                                                                                                        if (lot_size != '12632.4'):
                                                                                                            if (lot_size == '4181.0'):
                                                                                                                return 95000
                                                                                                            if (lot_size != '4181.0'):
                                                                                                                if (lot_size == '5445.0'):
                                                                                                                    return 97000
                                                                                                                if (lot_size != '5445.0'):
                                                                                                                    if (lot_size == '34412.4'):
                                                                                                                        return 499000
                                                                                                                    if (lot_size != '34412.4'):
                                                                                                                        if (lot_size == '17424.0'):
                                                                                                                            return 489000
                                                                                                                        if (lot_size != '17424.0'):
                                                                                                                            if (bedrooms == '4.0'):
                                                                                                                                if (lot_size == '12196.8'):
                                                                                                                                    return 433750
                                                                                                                                if (lot_size != '12196.8'):
                                                                                                                                    if (lot_size == '49658.4'):
                                                                                                                                        return 449900
                                                                                                                                    if (lot_size != '49658.4'):
                                                                                                                                        return 280810.58537
                                                                                                                            if (bedrooms != '4.0'):
                                                                                                                                if (lot_size == '9583.0'):
                                                                                                                                    return 539900
                                                                                                                                if (lot_size != '9583.0'):
                                                                                                                                    if (lot_size == '11325.6'):
                                                                                                                                        return 545000
                                                                                                                                    if (lot_size != '11325.6'):
                                                                                                                                        if (_type == 'Condo/Townhome/Row Home/Co-Op'):
                                                                                                                                            return 120000
                                                                                                                                        if (_type != 'Condo/Townhome/Row Home/Co-Op'):
                                                                                                                                            if (lot_size == '8276.0'):
                                                                                                                                                return 240698.57143
                                                                                                                                            if (lot_size != '8276.0'):
                                                                                                                                                return 307192.23669
                        if (term_matches(address, "address", "las") <= 0):
                            return 425238.23529
                if (full_bathrooms != '3.0'):
                    if (full_bathrooms == '5.0'):
                        return 557336.76923
                    if (full_bathrooms != '5.0'):
                        if (lot_size is None):
                            return 435592.79412
                        if (lot_size == '14374.8'):
                            return 859900
                        if (lot_size != '14374.8'):
                            if (lot_size == '23522.4'):
                                return 692500
                            if (lot_size != '23522.4'):
                                if (lot_size == '17424.0'):
                                    return 799000
                                if (lot_size != '17424.0'):
                                    if (address is None):
                                        return 422311.88776
                                    if (term_matches(address, "address", "rd") > 0):
                                        return 202475
                                    if (term_matches(address, "address", "rd") <= 0):
                                        if (lot_size == '44866.8'):
                                            return 699880
                                        if (lot_size != '44866.8'):
                                            if (term_matches(address, "address", "89122") > 0):
                                                return 251495
                                            if (term_matches(address, "address", "89122") <= 0):
                                                if (lot_size == '7840.0'):
                                                    return 312496
                                                if (lot_size != '7840.0'):
                                                    if (lot_size == '3484.0'):
                                                        return 297663
                                                    if (lot_size != '3484.0'):
                                                        if (lot_size == '5662.0'):
                                                            return 329000
                                                        if (lot_size != '5662.0'):
                                                            if (lot_size == '23086.8'):
                                                                return 282499.5
                                                            if (lot_size != '23086.8'):
                                                                if (lot_size == '6098.0'):
                                                                    return 365761.28571
                                                                if (lot_size != '6098.0'):
                                                                    if (lot_size == '8276.0'):
                                                                        return 290000
                                                                    if (lot_size != '8276.0'):
                                                                        if (lot_size == '87120.0'):
                                                                            return 305495
                                                                        if (lot_size != '87120.0'):
                                                                            if (term_matches(address, "address", "89129") > 0):
                                                                                return 348829.66667
                                                                            if (term_matches(address, "address", "89129") <= 0):
                                                                                if (lot_size == '15246.0'):
                                                                                    return 324500
                                                                                if (lot_size != '15246.0'):
                                                                                    if (term_matches(address, "address", "ave") > 0):
                                                                                        return 421983.25
                                                                                    if (term_matches(address, "address", "ave") <= 0):
                                                                                        if (bedrooms is None):
                                                                                            return 485464.73077
                                                                                        if (bedrooms == '3.0'):
                                                                                            return 269950
                                                                                        if (bedrooms != '3.0'):
                                                                                            if (term_matches(address, "address", "st") > 0):
                                                                                                return 434753.76923
                                                                                            if (term_matches(address, "address", "st") <= 0):
                                                                                                if (lot_size == '20037.6'):
                                                                                                    return 684450
                                                                                                if (lot_size != '20037.6'):
                                                                                                    return 498708.80556
            if (half_bathrooms != '0.0'):
                if (full_bathrooms == '3.0'):
                    if (half_bathrooms == '1.0'):
                        if (lot_size is None):
                            return 429786.72727
                        if (lot_size == '13939.2'):
                            return 761666.33333
                        if (lot_size != '13939.2'):
                            if (lot_size == '84070.8'):
                                return 989000
                            if (lot_size != '84070.8'):
                                if (lot_size == '1742.0'):
                                    return 179526
                                if (lot_size != '1742.0'):
                                    if (lot_size == '2178.0'):
                                        return 186160
                                    if (lot_size != '2178.0'):
                                        if (lot_size == '12196.8'):
                                            return 785000
                                        if (lot_size != '12196.8'):
                                            if (lot_size == '2613.0'):
                                                return 99999.5
                                            if (lot_size != '2613.0'):
                                                if (lot_size == '35719.2'):
                                                    return 898076
                                                if (lot_size != '35719.2'):
                                                    if (lot_size == '5662.0'):
                                                        return 310213.63636
                                                    if (lot_size != '5662.0'):
                                                        if (lot_size == '18730.8'):
                                                            return 717456
                                                        if (lot_size != '18730.8'):
                                                            if (lot_size == '20473.2'):
                                                                if (address is None):
                                                                    return 610132.6
                                                                if (term_matches(address, "address", "ct") > 0):
                                                                    return 712438
                                                                if (term_matches(address, "address", "ct") <= 0):
                                                                    return 541929
                                                            if (lot_size != '20473.2'):
                                                                if (lot_size == '4791.0'):
                                                                    return 220000
                                                                if (lot_size != '4791.0'):
                                                                    if (lot_size == '5227.0'):
                                                                        return 307541.42857
                                                                    if (lot_size != '5227.0'):
                                                                        if (lot_size == '49658.4'):
                                                                            return 775000
                                                                        if (lot_size != '49658.4'):
                                                                            if (lot_size == '30927.6'):
                                                                                return 749900
                                                                            if (lot_size != '30927.6'):
                                                                                if (lot_size == '3049.0'):
                                                                                    return 284640.5
                                                                                if (lot_size != '3049.0'):
                                                                                    if (lot_size == '46173.6'):
                                                                                        return 720000
                                                                                    if (lot_size != '46173.6'):
                                                                                        if (bedrooms is None):
                                                                                            return 437881.92169
                                                                                        if (bedrooms == '3.0'):
                                                                                            return 503154.41176
                                                                                        if (bedrooms != '3.0'):
                                                                                            if (lot_size == '22651.2'):
                                                                                                return 715000
                                                                                            if (lot_size != '22651.2'):
                                                                                                if (lot_size == '6098.0'):
                                                                                                    if (bedrooms == '6.0'):
                                                                                                        return 229200
                                                                                                    if (bedrooms != '6.0'):
                                                                                                        return 382976
                                                                                                if (lot_size != '6098.0'):
                                                                                                    if (lot_size == '17859.6'):
                                                                                                        return 670000
                                                                                                    if (lot_size != '17859.6'):
                                                                                                        if (lot_size == '15246.0'):
                                                                                                            return 660000
                                                                                                        if (lot_size != '15246.0'):
                                                                                                            if (lot_size == '10890.0'):
                                                                                                                return 591750
                                                                                                            if (lot_size != '10890.0'):
                                                                                                                if (lot_size == '19602.0'):
                                                                                                                    return 645812
                                                                                                                if (lot_size != '19602.0'):
                                                                                                                    if (address is None):
                                                                                                                        return 426739.376
                                                                                                                    if (term_matches(address, "address", "dr") > 0):
                                                                                                                        return 482396.91667
                                                                                                                    if (term_matches(address, "address", "dr") <= 0):
                                                                                                                        if (lot_size == '22215.6'):
                                                                                                                            return 625000
                                                                                                                        if (lot_size != '22215.6'):
                                                                                                                            if (lot_size == '11325.6'):
                                                                                                                                return 225000
                                                                                                                            if (lot_size != '11325.6'):
                                                                                                                                if (lot_size == '8276.0'):
                                                                                                                                    return 318933.33333
                                                                                                                                if (lot_size != '8276.0'):
                                                                                                                                    if (lot_size == '87120.0'):
                                                                                                                                        return 338490
                                                                                                                                    if (lot_size != '87120.0'):
                                                                                                                                        if (lot_size == '4356.0'):
                                                                                                                                            return 325515.66667
                                                                                                                                        if (lot_size != '4356.0'):
                                                                                                                                            if (lot_size == '17424.0'):
                                                                                                                                                return 260000
                                                                                                                                            if (lot_size != '17424.0'):
                                                                                                                                                return 431563.52
                    if (half_bathrooms != '1.0'):
                        return 696599.8
                if (full_bathrooms != '3.0'):
                    if (full_bathrooms == '4.0'):
                        if (lot_size is None):
                            return 558592.47222
                        if (lot_size == '10890.0'):
                            return 867999.66667
                        if (lot_size != '10890.0'):
                            if (lot_size == '14810.4'):
                                return 852933.33333
                            if (lot_size != '14810.4'):
                                if (lot_size == '91476.0'):
                                    return 1000000
                                if (lot_size != '91476.0'):
                                    if (lot_size == '20908.8'):
                                        return 975000
                                    if (lot_size != '20908.8'):
                                        if (address is None):
                                            return 531901.88
                                        if (term_matches(address, "address", "dr") > 0):
                                            return 655574.08333
                                        if (term_matches(address, "address", "dr") <= 0):
                                            if (lot_size == '7840.0'):
                                                return 295725
                                            if (lot_size != '7840.0'):
                                                if (bedrooms is None):
                                                    return 525480.94048
                                                if (bedrooms == '3.0'):
                                                    return 949000
                                                if (bedrooms != '3.0'):
                                                    if (lot_size == '6969.0'):
                                                        return 320725
                                                    if (lot_size != '6969.0'):
                                                        if (term_matches(address, "address", "89121") > 0):
                                                            return 180000
                                                        if (term_matches(address, "address", "89121") <= 0):
                                                            if (lot_size == '22651.2'):
                                                                return 865000
                                                            if (lot_size != '22651.2'):
                                                                if (lot_size == '46173.6'):
                                                                    return 849900
                                                                if (lot_size != '46173.6'):
                                                                    if (lot_size == '6098.0'):
                                                                        return 349666.66667
                                                                    if (lot_size != '6098.0'):
                                                                        if (lot_size == '6534.0'):
                                                                            return 400758
                                                                        if (lot_size != '6534.0'):
                                                                            if (term_matches(address, "address", "89129") > 0):
                                                                                return 401972
                                                                            if (term_matches(address, "address", "89129") <= 0):
                                                                                if (bedrooms == '4.0'):
                                                                                    return 613386.6
                                                                                if (bedrooms != '4.0'):
                                                                                    if (lot_size == '20037.6'):
                                                                                        return 658402.66667
                                                                                    if (lot_size != '20037.6'):
                                                                                        if (lot_size == '30927.6'):
                                                                                            return 749900
                                                                                        if (lot_size != '30927.6'):
                                                                                            if (lot_size == '10018.8'):
                                                                                                return 667995
                                                                                            if (lot_size != '10018.8'):
                                                                                                if (term_matches(address, "address", "rd") > 0):
                                                                                                    return 318351
                                                                                                if (term_matches(address, "address", "rd") <= 0):
                                                                                                    return 518396.97619
                    if (full_bathrooms != '4.0'):
                        return 722554.42308


def test_general():
    prediction=predict_price(full_bathrooms='2.0',bedrooms='2.0',_type= 'Condo/Townhome/Row Home/Co-Op')
    assert prediction == 182748.6719
    print('test done')

    
if __name__ == '__main__':
    test_general()
    print('begin')
    print(predict_price(full_bathrooms='2.0',bedrooms='2.0',_type= 'Condo/Townhome/Row Home/Co-Op'))
#    print(predict_price(bedrooms=0,size_sqft=100,full_bathrooms=2))


