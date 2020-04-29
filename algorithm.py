class K_Means:
    def __init__(self, k=2, centroid_selection = 1, tol=0.00001, max_iter=300):
        #k no of cluster
        self.k = k
        #used to decide if optimized or more iterations are required
        self.tol = tol
        #max no of iterations
        self.max_iter = max_iter
        # centroid selection strategy
        self.centroid_selection = centroid_selection
        #result centroids
        self.result_centroids = {}
        #feature index ~ cluster
        self.result_classifications = {}

    def fit(self,data):

        self.centroids = {}
        for i in range(self.k):
            #intialize centroids for k cluster, initialize first k data points
            #as centroids (it could be random also)         
            if self.centroid_selection == 1:
                self.centroids[i] = data[i]
            elif self.centroid_selection == 2:
                #print(2)
                self.centroids[i] = np.array([X[:,0].mean() + np.random.rand(), X[:,1].mean() + np.random.rand()]) 
                #self.centroids[i] = data[(X[:,0].size -1) - i]
            else :
                self.centroids[i] = data[np.random.randint(0, X[:,0].size)]
            
            #self.centroids[i] = data[i]
            #print(self.centroids)

        for i in range(self.max_iter):
            self.actual_iteration = i + 1
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            
            idx = 0
            for featureset in data:
                #Find the Euclidean distance of each point in the data set with the identified K points — cluster centroids.
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                #Assign each data point to the closest centroid using the distance found
                self.classifications[classification].append(featureset)
                self.result_classifications[idx] = classification
                idx = idx + 1

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                #Find the new centroid by taking the average of the points in each cluster group
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True
            #Repeat for a fixed number of iteration or till the centroids don’t change.
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol)
                #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                self.result_centroids[c] = current_centroid
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                #print("Optimized")
                break

                
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
