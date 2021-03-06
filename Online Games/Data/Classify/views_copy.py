for i in classifier:
    if i == 'Logistic Regression':
        t0 = time.clock()
        clf.append(clfa.fit(xtrain,ytrain))
        t1 = time.clock()
        yp = clfa.predict(xtest)
        t2 = time.clock()
        print(accuracy_score(ytest, yp))
        print(classification_report(ytest, yp))
        print(precision_recall_fscore_support(ytest, yp, average='weighted'))
        train_time.append("{:.2f}".format((t1 - t0) * 1000))
        test_time.append("{:.2f}".format((t2 - t1) * 1000))
        accr.append("{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100))
    elif i == 'Decision Tree':
        t0 = time.clock()
        clf.append(clfd.fit(xtrain,ytrain))
        t1 = time.clock()
        yp = clfd.predict(xtest)
        t2 = time.clock()
        print(accuracy_score(ytest, yp))
        print(classification_report(ytest, yp))
        print(precision_recall_fscore_support(ytest, yp, average='weighted'))
        train_time.append("{:.2f}".format((t1 - t0) * 1000))
        test_time.append("{:.2f}".format((t2 - t1) * 1000))
        accr.append("{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100))
    elif i == 'Support Vector Machine':
        t0 = time.clock()
        clf.append(clfb.fit(xtrain,ytrain))
        t1 = time.clock()
        yp = clfb.predict(xtest)
        t2 = time.clock()
        print(accuracy_score(ytest, yp))
        print(classification_report(ytest, yp))
        print(precision_recall_fscore_support(ytest, yp, average='weighted'))
        train_time.append("{:.2f}".format((t1 - t0) * 1000))
        test_time.append("{:.2f}".format((t2 - t1) * 1000))
        accr.append("{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100))
    elif i == 'RandomForest':
        t0 = time.clock()
        clf.append(clff.fit(xtrain,ytrain))
        t1 = time.clock()
        yp = clff.predict(xtest)
        t2 = time.clock()
        print(accuracy_score(ytest, yp))
        print(classification_report(ytest, yp))
        print(precision_recall_fscore_support(ytest, yp, average='weighted'))
        train_time.append("{:.2f}".format((t1 - t0) * 1000))
        test_time.append("{:.2f}".format((t2 - t1) * 1000))
        accr.append("{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100))
    elif i == 'XGBoost':
        t0 = time.clock()
        clf.append(clfc.fit(xtrain,ytrain))
        t1 = time.clock()
        yp = clfc.predict(xtest)
        t2 = time.clock()
        print(accuracy_score(ytest, yp))
        print(classification_report(ytest, yp))
        print(precision_recall_fscore_support(ytest, yp, average='weighted'))
        train_time.append("{:.2f}".format((t1 - t0) * 1000))
        test_time.append("{:.2f}".format((t2 - t1) * 1000))
        accr.append("{:.2f}".format((sum(ytest == yp) / float(len(yp))) * 100))