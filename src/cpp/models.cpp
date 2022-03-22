//
// Created by Tsimafei Prakapenka on 22.03.22.
//

#include "models.h"
#include "iostream"
#include "map"
#include "vector"
#include "fstream"
#include "string"
#include "filesystem"

using namespace std;

struct Phrase {
    string english;
    double logprob;

    Phrase(){};

    Phrase(const string &english_phrase, double phrase_logprob) : english(english_phrase), logprob(phrase_logprob) {};


    bool operator<(const Phrase &other_phrase) const {
        return (-logprob < -other_phrase.logprob);
    }
};

struct NgramStats{
    double backoff;
    double logprob;

    NgramStats(){};

    NgramStats(double backoff_logprob, double ngram_logprob) : backoff(backoff_logprob), logprob(ngram_logprob) {};
};

vector<string> split_line(string line, string delimiter) {
    vector<string> result;
    size_t pos_i = 0, pos_j = 0;

    while ((pos_j = line.find(delimiter, pos_i)) != string::npos) {
        result.push_back(line.substr(pos_i, pos_j - pos_i));
        pos_i = pos_j + delimiter.length();
    }
    result.push_back(line.substr(pos_i));
    return result;
}

class TM {
    map<string, vector<Phrase>> tm;

    void init_tm(string source_filename){
        std::ifstream file(source_filename);
        string line, delimiter = " ||| ";
        while (std::getline(file, line)) {
            vector<string> tokens = split_line(line, delimiter);
            string french_phrase = tokens[0], english_phrase = tokens[1];
            double logprob = stod(tokens[2]);

            auto it = tm.find(french_phrase);

            if (it != tm.end()) {
                tm[french_phrase].push_back(Phrase(english_phrase, logprob));
            } else {
                tm.insert({french_phrase, {Phrase(english_phrase, logprob)}});
            }
        }
    }
    void prune_tm(int k){
        for (auto iter = tm.begin(); iter != tm.end(); ++iter) {
            int pruned_size = k < iter->second.size() ? k : iter->second.size();
            vector<Phrase> pruned_translations(pruned_size);
            partial_sort_copy(iter->second.begin(), iter->second.end(),
                              pruned_translations.begin(), pruned_translations.end());
            tm[iter->first] = pruned_translations;
        }
    }
public:
    TM(string source_filename, int k) : tm() {
        init_tm(source_filename);
        prune_tm(k);
    }
};

class LM {

};

int main(void) {
    auto tm = TM("../data/tm", 10);
    return 0;
}