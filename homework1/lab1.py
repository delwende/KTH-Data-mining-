
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import sys
import os.path
import string
import os
import re

import time
import binascii
import numpy as np
from time import clock
from random import randint, seed, choice, random
import string
import itertools


# In[2]:


documents = []
printedbodies = {}
data=''


# ### Class shingling  

# In[3]:


class Shingling:
    #global documents , printedbodies
    d = {}

    t = {}
    docsAsShingleSets = {}

    docNames = []

    totalShingles = 0
    shingleNo = 0
    shingle_size=0
    def __init__(self,documents):
        i = 0
        self.documents=documents
        for value in self.documents:


            # create a dictionary where key=docid and value=document text
            self.d[i] = value
            # split text into words
            self.d[i] = re.sub("[^\w]", " ", self.d[i]).split()

            # remove rows with empty values from dictionary d
            if self.d[i]:
                i = i + 1
            else:
                del self.d[i]
                del self.body[i]

        # =============================================================================
        #               Convert Documents To Sets of Shingles
        # =============================================================================

      
        ######Ask user to give a value for k
        while True:
            try:
                self.shingle_size = int(raw_input("Please enter k value for k-shingles: "))
            except ValueError:
                print("Your input is not valid. Give a positive natural number > 0...")
                continue
            if self.shingle_size <= 0:
                continue
            else:
                break
    def func_Shingling(self):
        print "Shingling articles..."

        t0 = time.time()
        # loop through all the documents
        for i in range(0, len(self.d)):

            # Read all of the words (they are all on one line)
            words = self.d[i]

            # Retrieve the article ID
            docID = i

            # Maintain a list of all document IDs.
            self.docNames.append(docID)

            # 'shinglesInDoc' will hold all of the unique shingles present in the
            # current document. If a shingle ID occurs multiple times in the document,
            # it will only appear once in the set.

            # keep word shingles
            self.shinglesInDocWords = set()

            # keep hashed shingles
            self.shinglesInDocInts = set()

            shingle = []
            # For each word in the document...
            for index in range(len(words) - self.shingle_size + 1):
                # Construct the shingle text by combining k words together.
                shingle = words[index:index + self.shingle_size]
                shingle = ' '.join(shingle)

                # Hash the shingle to a 32-bit integer.
                crc = binascii.crc32(shingle) & 0xffffffff

                if shingle not in self.shinglesInDocWords:
                    self.shinglesInDocWords.add(shingle)
                # Add the hash value to the list of shingles for the current document.
                # Note that set objects will only add the value to the set if the set
                # doesn't already contain it.

                if crc not in self.shinglesInDocInts:
                    self.shinglesInDocInts.add(crc)
                    # Count the number of shingles across all documents.
                    self.shingleNo = self.shingleNo + 1
                else:
                    del shingle
                    index = index - 1

            # Store the completed list of shingles for this document in the dictionary.
            self.docsAsShingleSets[docID] = self.shinglesInDocInts

        self.totalShingles = self.shingleNo

        print 'Total Number of Shingles', self.shingleNo
        # Report how long shingling took.
        print '\nShingling ' + str(len(self.docsAsShingleSets)) + ' docs took %.2f sec.' % (time.time() - t0)

        print '\nAverage shingles per doc: %.2f' % (self.shingleNo / len(self.docsAsShingleSets))


# ### Minhashing
# 

# In[4]:





class MinHashing:
    
   
    
    
    numHashes=0
    numDocs=0
   
    signatures = []
    def __init__(self):
       
       
        ######Ask user to give a value for hash functions to be used
        while True:
            try:
                self.numHashes = int(raw_input("\nPlease enter how many hash functions you want to be used: "))
            except ValueError:
                print("Your input is not valid. Give a positive natural number > 0...")
                continue
            if self.numHashes <= 0:
                continue
            else:
                break

        print '\nGenerating random hash functions...'

    # =============================================================================
    #                 Generate MinHash Signatures
    # =============================================================================
  

    

    # https://www.codeproject.com/Articles/691200/Primality-test-algorithms-Prime-test-The-fastest-w
    # check if integer n is a prime
    # probabilistic method, much faster than usual priminality tests
    def MillerRabinPrimalityTest(self,number):
        '''
        because the algorithm input is ODD number than if we get
        even and it is the number 2 we return TRUE ( spcial case )
        if we get the number 1 we return false and any other even
        number we will return false.
        '''
        if number == 2:
            return True
        elif number == 1 or number % 2 == 0:
            return False

        ''' first we want to express n as : 2^s * r ( were r is odd ) '''

        ''' the odd part of the number '''
        oddPartOfNumber = number - 1

        ''' The number of time that the number is divided by two '''
        timesTwoDividNumber = 0

        ''' while r is even divid by 2 to find the odd part '''
        while oddPartOfNumber % 2 == 0:
            oddPartOfNumber = oddPartOfNumber / 2
            timesTwoDividNumber = timesTwoDividNumber + 1

        '''
        since there are number that are cases of "strong liar" we
        need to check more then one number
        '''
        for time in range(3):

            ''' choose "Good" random number '''
            while True:
                ''' Draw a RANDOM number in range of number ( Z_number )  '''
                randomNumber = randint(2, number) - 1
                if randomNumber != 0 and randomNumber != 1:
                    break

            ''' randomNumberWithPower = randomNumber^oddPartOfNumber mod number '''
            print randomNumber, oddPartOfNumber, number
            randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

            ''' if random number is not 1 and not -1 ( in mod n ) '''
            if (randomNumberWithPower != 1) and (randomNumberWithPower != number - 1):
                # number of iteration
                iterationNumber = 1

                ''' while we can squre the number and the squered number is not -1 mod number'''
                while (iterationNumber <= timesTwoDividNumber - 1) and (randomNumberWithPower != number - 1):
                    ''' squre the number '''
                    randomNumberWithPower = pow(randomNumberWithPower, 2, number)

                    # inc the number of iteration
                    iterationNumber = iterationNumber + 1
                '''
                if x != -1 mod number then it because we did not found strong witnesses
                hence 1 have more then two roots in mod n ==>
                n is composite ==> return false for primality
                '''
                if (randomNumberWithPower != (number - 1)):
                    return False

        ''' well the number pass the tests ==> it is probably prime ==> return true for primality '''
        return True
    # Our random hash function will take the form of:
    #   h(x) = (a*x + b) % c
    # Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
    # a prime number just greater than shingleNo.

    # Generate a list of 'k' random coefficients for the random hash functions,
    # while ensuring that the same value does not appear multiple times in the
    # list.
    def pickRandomCoeffs(self, k, maxShingleID):
        # Create a list of 'k' random values.
        randList = []

        while k > 0:
            # Get a random shingle ID.
            randIndex = randint(0, maxShingleID)

            # Ensure that each random number is unique.
            while randIndex in randList:
                randIndex = randint(0, maxShingleID)

                # Add the random number to the list.
            randList.append(randIndex)
            k = k - 1

        return randList

    def generatHash(self,docNames,shingleNo,docsAsShingleSets):
        # Time this step.
        t0 = time.time()
        self.docNames=docNames
        self.shingleNo=shingleNo
        self.docsAsShingleSets= docsAsShingleSets
        # Record the total number of shingles
        i = 1
        # find first prime which is higher than the total number of shingles
        # print 'Total number of shingles = ', shingleNo
        while not self.MillerRabinPrimalityTest(self.shingleNo + i):
            i = i + 1
        print 'Next prime = ', self.shingleNo + i

        maxShingleID = self.shingleNo
        nextPrime = self.shingleNo + i


   

        # For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.
        coeffA = self.pickRandomCoeffs(self.numHashes,maxShingleID)
        coeffB = self.pickRandomCoeffs(self.numHashes,maxShingleID)

        print '\nGenerating MinHash signatures for all documents...'

        # List of documents represented as signature vectors
        

        # Rather than generating a random permutation of all possible shingles,
        # we'll just hash the IDs of the shingles that are *actually in the document*,
        # then take the lowest resulting hash code value. This corresponds to the index
        # of the first shingle that you would have encountered in the random order.
        # For each document...
        for docID in self.docNames:

            # Get the shingle set for this document.
            shingleIDSet = self.docsAsShingleSets[docID]

            # The resulting minhash signature for this document.
            signature = []

            # For each of the random hash functions...
            for i in range(0, self.numHashes):

                # For each of the shingles actually in the document, calculate its hash code
                # using hash function 'i'.

                # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
                # the maximum possible value output by the hash.
                minHashCode = nextPrime + 1

                # For each shingle in the document...
                for shingleID in shingleIDSet:
                    # Evaluate the hash function.
                    hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime

                    # Track the lowest hash code seen.
                    if hashCode < minHashCode:
                        minHashCode = hashCode

                # Add the smallest hash code value as component number 'i' of the signature.
                signature.append(minHashCode)

            # Store the MinHash signature for this document.
            self.signatures.append(signature)

        # Calculate the elapsed time (in seconds)
        elapsed = (time.time() - t0)

        print "\nGenerating MinHash signatures took %.2fsec" % elapsed
        self.numDocs = len(self.signatures)


# ### CompareSets

# In[5]:



from decimal import *
class CompareSets:
    ######Ask user to choose a document
    docid=0
    neighbors=0
    numDocs=0
    fp = []
    tp = []
    def __init__(self,numDocs):
        self.numDocs=numDocs
        while True:
            try:
                self.docid = int(raw_input(
                    "Please enter the document id you are interested in. The valid document ids are 1 - " + str(
                        self.numDocs) + ": "))
            except ValueError:
                print("Your input is not valid.")
                continue
            if self.docid <= 0 or self.docid > self.numDocs:
                print ("Your input is out of the defined range...")
                continue
            else:
                break

        ######Ask user to give desired number of neighbors
        while True:
            try:
                self.neighbors = int(raw_input("Please enter the number of closest neighbors you want to find... "))
            except ValueError:
                print("Your input is not valid.")
                continue
            if self.neighbors <= 0:
                continue
            else:
                break
    # Define a function to map a 2D matrix coordinate into a 1D index.
    def getTriangleIndex(self,i, j,docsAsShingleSets):
        # If i == j that's an error.
        if i == j:
            sys.stderr.write("Can't access triangle matrix with i == j")
            sys.exit(1)
        # If j < i just swap the values.
        if j < i:
            temp = i
            i = j
            j = temp

        # Calculate the index within the triangular array.
        # This fancy indexing scheme is taken from pg. 211 of:
        # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
        # But I adapted it for a 0-based index.
        # Note: The division by two should not truncate, it
        #       needs to be a float.
        k = int(i * (len(docsAsShingleSets) - (i + 1) / 2.0) + j - i) - 1

        return k

    def JaccardSimilarities(self,docNames,shingleNo,docsAsShingleSets):
        
        
        numElems = int(len(docsAsShingleSets) * (len(docsAsShingleSets) - 1) / 2)

        # Initialize two empty lists to store the similarity values.
        # 'JSim' will be for the actual Jaccard Similarity values.
        # 'estJSim' will be for the estimated Jaccard Similarities found by comparing
        # the MinHash signatures.
        print numElems
        JSim = np.zeros(numElems)#[0 for x in range(numElems)]
       # estJSim = np.zeros(numElems)#[0 for x in range(numElems)]


        # =============================================================================
        #                 Calculate Jaccard Similarities
        # =============================================================================
        # In this section, we will directly calculate the Jaccard similarities by
        # comparing the sets. This is included here to show how much slower it is than
        # the MinHash approach.

        # Calculating the Jaccard similarities gets really slow for large numbers
        # of documents.
       

       

        print "\nCalculating Jaccard Similarities of Shingles..."

        # Time the calculation.
        t0 = time.time()

        s0 = len(docsAsShingleSets[0])
        # For every document pair...
        i = self.docid

        # Print progress every 100 documents.
        if (i % 100) == 0:
            print "  (" + str(i) + " / " + str(len(docsAsShingleSets)) + ")"

        # Retrieve the set of shingles for document i.
        s1 = docsAsShingleSets[docNames[i]]
        neighbors_of_given_documentSHINGLES = {}
       

        for j in range(0, len(docsAsShingleSets)):
            if j != i:
                # Retrieve the set of shingles for document j.
                s2 = docsAsShingleSets[docNames[j]]

                # Calculate and store the actual Jaccard similarity.
                JSim[self.getTriangleIndex(i, j,docsAsShingleSets)] = (len(s1.intersection(s2)) / float(len(s1.union(s2))))
                percsimilarity = JSim[self.getTriangleIndex(i, j,docsAsShingleSets)] * 100
                if (percsimilarity > 0):
                    # Print out the match and similarity values with pretty spacing.
                    #print "  %5s --> %5s   %.2f%s   " % (docNames[i], docNames[j], percsimilarity, '%')
                    neighbors_of_given_documentSHINGLES[j] = percsimilarity

        sorted_neigborsSHINGLES = sorted(neighbors_of_given_documentSHINGLES.items(), key=lambda x: x[1], reverse=True)

        print 'Comparing Shingles ...'
        print "The " + str(self.neighbors) + " closest neighbors of document " + str(docNames[i]) + " are:"
        for i in range(0, self.neighbors):
            if i >= len(sorted_neigborsSHINGLES):
                break
            self.tp.append(sorted_neigborsSHINGLES[i][0])
            print "Shingles of Document " + str(sorted_neigborsSHINGLES[i][0]) + " with Jaccard Similarity " + str(
                round(sorted_neigborsSHINGLES[i][1], 2)) + "%"

            # Calculate the elapsed time (in seconds)
        elapsed = (time.time() - t0)

        print 'These are the True Positives, since no time saving assumptions were made while calculating the Jaccard similarity of shingles'
        print "\nCalculating all Jaccard Similarities of Shingles Took %.2fsec" % elapsed
        print '\nNote: In this section, we directly calculated the Jaccard similarities by comparing the shingle sets. This is included here to show how much slower it is than the MinHash and LSH approach.'
        print '\nMoreover, the similarities calculated above are the actual similarities of the documents, since there were no assumption made'

        # shingleNo =  shingleNo + s0
        # print 'number', shingleNo

        # Delete the Jaccard Similarities, since it's a pretty big matrix.
        del JSim





# ### CompareSignatures

# In[6]:


class CompareSignatures:
    # =============================================================================
    #                  Compare ALl Signatures & Display Similar Document Pairs
    # =============================================================================
    tpsig = 0
    fpsig = 0

    

    threshold = 0
    def __init__(self,signatures):
        print 'Number of signatures', len(signatures)
        # Count the true positives and false positives.

        print '\nNow we will calculate Jaccard Similarity between signatures'
        print "Values shown are the estimated Jaccard similarity"
    # Define a function to map a 2D matrix coordinate into a 1D index.
    def getTriangleIndex(self,i, j,docsAsShingleSets):
        # If i == j that's an error.
        if i == j:
            sys.stderr.write("Can't access triangle matrix with i == j")
            sys.exit(1)
        # If j < i just swap the values.
        if j < i:
            temp = i
            i = j
            j = temp

        # Calculate the index within the triangular array.
        # This fancy indexing scheme is taken from pg. 211 of:
        # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
        # But I adapted it for a 0-based index.
        # Note: The division by two should not truncate, it
        #       needs to be a float.
        k = int(i * (len(docsAsShingleSets) - (i + 1) / 2.0) + j - i) - 1

        return k

    def computefunct(self,docNames,docid,signatures,numDocs,numHashes,tp,docsAsShingleSets,neighbors):
        t0 = time.time()
        # For each of the document pairs...
        # for i in range(1, numDocs-1):
        i = docid
        signature1 = signatures[i]

        neighbors_of_given_documentSIGNATURES = {}
        # Calculate the number of elements needed in our triangle matrix
        numElems = int(len(docsAsShingleSets) * (len(docsAsShingleSets) - 1) / 2)
        estJSim = np.zeros(numElems)#[0 for x in range(numElems)]
        for j in range(0, numDocs):
            if (i != j):
                signature2 = signatures[j]
                count = 0
                # Count the number of positions in the minhash signature which are equal.
                for k in range(0, numHashes):

                    if (signature1[k] == signature2[k]):
                        count = count + 1

                # Record the percentage of positions which matched.
                estJSim[self.getTriangleIndex(i, j,docsAsShingleSets)] = (count / float(numHashes))

                # Retrieve the estimated similarity value for this pair.
                # estJ = float(estJSim[getTriangleIndex(i, j)])

                # If the similarity is above the threshold...
                if float(estJSim[self.getTriangleIndex(i, j,docsAsShingleSets)]) > 0:

                    # Calculate the actual Jaccard similarity for validation.
                    s1 = set(signature1)
                    s2 = set(signature2)

                    J = len(s1.intersection(s2)) / float(len(s1.union(s2)))
                    neighbors1 = []
                    if (float(J) > self.threshold):
                        percsimilarity = estJSim[self.getTriangleIndex(i, j,docsAsShingleSets)] * 100

                        percJ = J * 100
                        # Print out the match and similarity values with pretty spacing.
                        # print "  %5s --> %5s   %.2f%s " % (docNames[i], docNames[j], percJ, '%')
                        neighbors_of_given_documentSIGNATURES[j] = percJ

        sorted_neigborsSIGNATURES = sorted(neighbors_of_given_documentSIGNATURES.items(), key=lambda x: x[1], reverse=True)
        # print "Sorted Neighbors Signatures", sorted_neigbors, "%"

        sigpos = []
        print 'Comparing Signatures...'
        print "The " + str(neighbors) + " closest neighbors of document " + str(docNames[i]) + " are:"
        for i in range(0, neighbors):
            if i >= len(sorted_neigborsSIGNATURES):
                break
            print "Signatures of Document " + str(sorted_neigborsSIGNATURES[i][0]) + " with Jaccard Similarity " + str(
                round(sorted_neigborsSIGNATURES[i][1], 2)) + "%"
            sigpos.append(sorted_neigborsSIGNATURES[i][0])

        fpsig = neighbors - len(list(set(tp).intersection(sigpos)))
        tpsig = neighbors - fpsig
        elapsed = (time.time() - t0)
        print '\n', tpsig, '/', neighbors, 'True Positives and', fpsig, '/', neighbors, 'False Positives Produced While Comparing Signatures',

        print "\nCalculating Jaccard Similarity of Signatures took %.2fsec" % elapsed



# ### LSH

# In[7]:


class LSH:
    threshold=0
    tp=[]
    docid=0
    band_size=0
    numHashes=0
    
    def __init__(self,numHashes,tp,docid):
        self.tp=tp
        self.docid=docid
        self.numHashes=numHashes
        while True:
            try:
                self.band_size = int(
                    raw_input("\nPlease enter the size of the band. Valid band rows are 1 - " + str(self.numHashes) + ": "))
            except ValueError:
                print("Your input is not valid.")
                continue
            if self.band_size <= 0 or self.band_size > self.numHashes:
                print ("Your input is out of the defined range...")
                continue
            else:
                break
    def get_band_hashes(self,minhash_row, band_size):
        band_hashes = []
        for i in range(len(minhash_row)):
            if i % band_size == 0:
                if i > 0:
                    band_hashes.append(band_hash)
                band_hash = 0
            band_hash += hash(minhash_row[i])
        return band_hashes
    def get_similar_docs(self,neighbors,docs, shingles, threshold, n_hashes, band_size,printedbodies, collectIndexes=True):
        t0 = time.time()
        lshsignatures = {}
        hash_bands = {}
        neighbors_of_given_documentLSH = {}
        random_strings = [str(random()) for _ in range(n_hashes)]
        docNum = 0

        # for key, value in t.iteritems():
        #    temp = [key, doc]
        #   tlist.append(temp)
        w = 0
        # for doc in docs.iteritems():
        for doc in docs:

            lshsignatures[w] = doc
            # shingles = generate_shingles(doc, shingle_size)
            # print 'doc', doc
            # shingles = doc

            minhash_row = doc
            # print 'minhash_row', minhash_row, type(minhash_row)
            band_hashes = self.get_band_hashes(minhash_row, band_size)
            # print 'band_hashes', band_hashes
            w = w + 1
            docMember = docNum if collectIndexes else doc
            for i in range(len(band_hashes)):
                if i not in hash_bands:
                    hash_bands[i] = {}
                if band_hashes[i] not in hash_bands[i]:
                    hash_bands[i][band_hashes[i]] = [docMember]
                else:
                    hash_bands[i][band_hashes[i]].append(docMember)
            docNum += 1

        similar_docs = set()
        similarity1 = []
        noPairs = 0
        print 'Comparing Signatures Found in the Same Buckets During LSH ...'
        # print "\n    Jaccard similarity After LSH\n"
        # print "    Pairs          Similarity"
        samebucketLSH = []
        samebucketcnt = 0
        for i in hash_bands:
            for hash_num in hash_bands[i]:
                if len(hash_bands[i][hash_num]) > 1:
                    for pair in itertools.combinations(hash_bands[i][hash_num], r=2):
                        if pair not in similar_docs:
                            similar_docs.add(pair)
                            if pair[0] == self.docid and pair[1] != self.docid:

                                s1 = set(lshsignatures[pair[0]])
                                s2 = set(lshsignatures[pair[1]])

                                sim = len(s1.intersection(s2)) / float(len(s1.union(s2)))
                                if (float(sim) > threshold):
                                    percsim = sim * 100
                                    # print  "  %5s --> %5s   %.2f%s" % (pair[0], pair[1], percsim,'%')
                                    noPairs = noPairs + 1
                                    # return similar texts

                                    # print 'TEXT WITH ID: ', pair[0], '\n AND BODY: ', body[pair[0]], '\n IS ', sim*100, '% SIMILAR TO', '\n TEXT WITH ID: ', pair[1], '\n AND BODY: ', body[pair[1]], '\n'
                                else:
                                    percsim = 0
                                neighbors_of_given_documentLSH[pair[1]] = percsim
                                samebucketLSH.append(pair[1])
                                samebucketcnt = samebucketcnt + 1
                                elapsed = (time.time() - t0)

        print 'Number of false positives while comparing signatures which were found in the same bucket',
        sorted_neigborsLSH = sorted(neighbors_of_given_documentLSH.items(), key=lambda x: x[1], reverse=True)
        # print "Sorted Neighbors Signatures", sorted_neigbors, "%"

        lshpos = []
        print 'Comparing Signatures Found in the Same Bucket During LSH...'
        print "The " + str(neighbors) + " closest neighbors of document " + str(self.docid) + " are:"
        for i in range(0, neighbors):
            if i >= len(sorted_neigborsLSH):
                break
            if sorted_neigborsLSH[i][1] > 0:
                print "\nChosen Signatures (After LSH) of Document " + str(sorted_neigborsLSH[i][0]) + " with Jaccard Similarity " + str(round(sorted_neigborsLSH[i][1], 2)) + "%"
               # print "\nBody of document " + str(sorted_neigborsLSH[i][0]) + "\n" + str(printedbodies[sorted_neigborsLSH[i][0]])
                lshpos.append(sorted_neigborsLSH[i][0])

        neighborsfplsh = neighbors - len(list(set(self.tp).intersection(lshpos)))
        neighborstplsh = neighbors - neighborsfplsh
        # totalfplsh =
        totaltplsh = len(list(set(self.tp).intersection(samebucketLSH)))
        totalfplsh = samebucketcnt - totaltplsh

        print '\nEvaluating the', neighbors, 'neighbors produced by LSH...'
        print neighborstplsh, 'out of', neighbors, 'TP and', neighborsfplsh, 'out of', neighbors, 'FP'
        print '\nEvaluating the', samebucketcnt, 'pairs which fell in the same bucket...'

        if samebucketcnt > 0:
            prctpLSH = round((totaltplsh / float(samebucketcnt)) * 100, 2)
            prcfpLSH = 100 - prctpLSH
            print totaltplsh, 'out of', samebucketcnt, 'documents which fell in the same bucket are TP', prctpLSH, '%'
            print totalfplsh, 'out of', samebucketcnt, 'documents which fell in the same bucket are FP', prcfpLSH, '%'
        else:
            print totaltplsh, 'out of', samebucketcnt, 'documents which fell in the same bucket are TP'
            print totalfplsh, 'out of', samebucketcnt, 'documents which fell in the same bucket are FP'

        return similar_docs


    def computefunt(self,neighbors,signatures,docsAsShingleSets,printedbodies,threshold):
        t0 = time.time()

       
        
        

        n_hashes = self.numHashes

        n_similar_docs = 2
        seed(42)

        finalshingles = docsAsShingleSets

        similar_docs = self.get_similar_docs(neighbors,signatures, finalshingles, threshold, n_hashes, self.band_size,printedbodies, collectIndexes=True)

        print '\nLocality Sensitive Hashing ' + str(len(signatures)) + ' docs took %.2f sec.' % (time.time() - t0)


        r = float(n_hashes / self.band_size)
        similarity = (1 / r) ** (1 / float(self.band_size))


# ### reading and cleaning file

# In[8]:


def readfile():
    global data
    print ('Reading files')
    print ('Please wait...')
    t0 = time.time()
    data=''
   

    for file in os.listdir("data1/"):
        if file.endswith(".sgm"):
            filename = os.path.join("data1", file)

            f = open(filename, 'r')
            data = data + f.read()

    print ('Reading data took %.2f sec.' % (time.time() - t0))

def preprocessing():
    global documents , printedbodies
    print ('Transforming data...')
    t0 = time.time()
    soup = BeautifulSoup(data, "html.parser")
    bodies = soup.findAll('body')
    i = 0
    for body in bodies:
        printedbodies[i] = body
        documents.append(
             re.sub(' +', ' ', str(body).replace("<body>", "").replace("</body>", "").translate(None, string.punctuation)
               .replace("", "").replace("\n", " ").lower()))
        i = i + 1

    print ('Transforming data took %.2f sec.' % (time.time() - t0))

    print ('The number of documents read was: ' + str(len(documents)))


# ### Main function

# In[9]:



def main():
    global documents , printedbodies
    
    readfile()
    
    preprocessing()
    single=Shingling(documents)
    single.func_Shingling()
    minhash=MinHashing()
    minhash.generatHash(single.docNames,single.shingleNo,single.docsAsShingleSets)
    compareSets=CompareSets(minhash.numDocs)
    compareSets.JaccardSimilarities(single.docNames,single.shingleNo,single.docsAsShingleSets)
    compareSignatures=CompareSignatures(minhash.signatures)
    compareSignatures.computefunct(single.docNames,compareSets.docid,minhash.signatures,minhash.numDocs,minhash.numHashes,compareSets.tp,single.docsAsShingleSets,compareSets.neighbors)
    lsh=LSH(minhash.numHashes,compareSets.tp,compareSets.docid)
    lsh.computefunt(compareSets.neighbors,minhash.signatures,single.docsAsShingleSets,printedbodies,0)
if __name__ == '__main__':
    main()

