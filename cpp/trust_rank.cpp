#include <iostream>
#include <fstream>
#include <list>
#include <iterator>
#include <map>
#include <omp.h>
#include <time.h>
#include <set>
#include <algorithm>
#include <vector>

using namespace std;

struct Page {
    list <int> ids_in; //in-pages
    int num_ids_in; //number of in-pages
    int num_ids_out; ////number of out-pages
    float score; //page-rank score
    float score_new; //used as temp variable to update the score
    Page(){
        num_ids_in = 0;
        num_ids_out = 0;
        score = 0.0;
    }
};

//show top ranked nodes
void show_top_nodes(std::set<std::pair<float, int>, std::greater<std::pair<float, int>>> g, int k=10){
    int i=0;
    for(auto& x : g){
        if(i==k)
            break;
        std::cout <<i+1<<") "<< x.second<<" "<<x.first<<"\n";
        i++;
    }
}

#define iterations 20
#define damping 0.8
#define n_threads 8
#define top_k 0

int main(int argc, char** argv){
    std::string input_file = argv[1];
    /*int iterations = atoi(argv[1]);
    int top_k = atoi(argv[2]);
    int n_threads = atoi(argv[3]);*/
    //Set thread count
    omp_set_num_threads(n_threads);
    cout<<"Input: "<<input_file<<endl;
    cout.setf(ios::fixed);
    cout.precision(3);
    ifstream cin(input_file);
    size_t lastindex = input_file.find_last_of(".");
    string name_file = input_file.substr(0, lastindex);
    int num_pages = 0;
    int num_edges = 0;
    int id, out;
    map <int, Page> pages;
    map <int, int> lookup;
    cout<<"Loading graph data..."<<endl;
    clock_t tStart = clock();
    while (cin>>id){
        cin>>out;
        if(!pages.count(id)){
            pages[id] = Page();
            lookup[num_pages]=id;
            num_pages++;
        }
        if(!pages.count(out)){
            pages[out] = Page();
            lookup[num_pages]=out;
            num_pages++;
        }
        pages[out].num_ids_in++;
        pages[out].ids_in.push_back(id);
        pages[id].num_ids_out++;
        num_edges++;
    }
    cout<<"Graph data loaded in "<<(double)(clock() - tStart)/CLOCKS_PER_SEC<<"s"<<endl;
    cout<<"Loaded "<<num_pages<<" nodes and "<<num_edges<<" edges"<<endl;

    // Initialize a vector with 100 integers of value 0
    std::vector<int> random_nodes(100);
    std::vector<int> chosen_nodes;
    // Generate 10 random numbers by lambda func and fill it in vector
    std::generate(random_nodes.begin(), random_nodes.end(), [chosen_nodes, num_pages]() {
        int r;
        do{
            r = rand() % num_pages;
        }
        while(std::find(chosen_nodes.begin(), chosen_nodes.end(), r) != chosen_nodes.end());
        return r;
    });

    // initialize trustRank
    float equal_prob = 1.0 / 100;
    for(auto& x : pages)
        if(std::find(random_nodes.begin(), random_nodes.end(), x.first) != random_nodes.end())
            x.second.score = equal_prob;
        else
            x.second.score = 0;

    float broadcastScore;

    cout<<"Begin algorithm..."<<endl;
    clock_t aStart = clock();
    for(int j=0;j<iterations;j++){
        clock_t tStart = clock();
        broadcastScore = 0.0;

        // evaluate the score for each node
        #pragma omp parallel for reduction(+: broadcastScore) schedule (dynamic, 32)
        for(int i=0;i<pages.size();i++){

            // find the ID corresponding to the i-page
            int idx = lookup[i];

            // used to store the new score
            pages[idx].score_new = 0.0;

            // if the node has no outgoing edges, then add its value to the broadcast score
            if(!pages[idx].num_ids_out)
                broadcastScore += pages[idx].score;

            // iterate over all the vertices with an incoming edge to compute the new value of the vertices
            if(pages[idx].num_ids_in>0){
                for(list <int> :: iterator it = pages[idx].ids_in.begin(); it != pages[idx].ids_in.end(); ++it)
                    pages[idx].score_new += pages[*it].score / pages[*it].num_ids_out;
            }

            // apply pageRank equation
            pages[idx].score_new = damping * pages[idx].score_new + (1.0 - damping) / num_pages;
        }

        // update broadcast score
        broadcastScore = damping * broadcastScore / num_pages;
        for(auto& x : pages){

            // add the global broadcast score to each edge to compute its final score
            x.second.score = x.second.score_new + broadcastScore;
        }
        cout<<"Iteration "<<j<<" completed in "<<(double)(clock() - tStart)/CLOCKS_PER_SEC<<"s"<<endl;
    }
    cout<<"Algorithm terminated in "<<(double)(clock() - aStart)/CLOCKS_PER_SEC<<"s"<<endl;

    //order ranked nodes
	std::set<std::pair<float, int>, std::greater<std::pair<float, int>>> ordered_pages;
	for (auto const &kv : pages)
    ordered_pages.emplace(kv.second.score, kv.first);

    string scope = to_string(top_k);
    if(top_k==0) scope = string("all");
    string result_path = string(name_file) + string("_trustrank_top-") + string(scope) + string(".txt");
    ofstream result_file(result_path);
    result_file.setf(ios::fixed);
    result_file.precision(9);

    clock_t rStart = clock();
    cout<<"Writing result..."<<endl;

    // write to k results, write all nodes if k==0
    int i=1;
    for(auto& x : ordered_pages){
        result_file <<x.second<<" "<<x.first<<"\n";
        if(i==top_k)
            break;
        i++;
    }

    string white_list_path = string(name_file) + string("_white_list.txt");
    ofstream white_list(white_list_path);
    for(auto& x : random_nodes){
        white_list <<x<<"\n";
    }

    cout<<"Results written in "<<(double)(clock() - rStart)/CLOCKS_PER_SEC<<"s"<<endl;
    result_file.close();

    cout.precision(9);
    cout<<"Top-"<<max(10, top_k)<<" nodes"<<endl;
    show_top_nodes(ordered_pages, max(10, top_k));
}
