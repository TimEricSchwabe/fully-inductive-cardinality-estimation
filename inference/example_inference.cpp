/**
 * FICE C++ Inference Example
 *
 * Demonstrates how to:
 *   1. Load the exported TorchScript query GNN model
 *   2. Memory-map precomputed entity/relation embeddings
 *   3. Parse a simplified SPARQL BGP query string
 *   4. Build the query graph tensors
 *   5. Run the query GNN and return a cardinality estimate
 *
 * Build:
 *   mkdir build && cd build
 *   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
 *   make
 *
 * Run:
 *   ./fice_inference /path/to/inference_output "?x <rel1> <entity1> . ?x <rel2> ?y"
 */

#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Minimal JSON parser (for metadata.json and uri_to_id.json)
// Only handles the subset we need: flat objects, nested objects, strings, ints,
// floats, bools. No arrays. For production use nlohmann/json instead.
// ---------------------------------------------------------------------------

struct JsonValue {
    using ObjMap = std::unordered_map<std::string, JsonValue>;

    enum Type { STRING, NUMBER, BOOL, OBJECT, NONE } type = NONE;
    std::string str_val;
    double num_val = 0;
    bool bool_val = false;
    std::unique_ptr<ObjMap> obj_val;

    JsonValue() = default;
    JsonValue(JsonValue&&) = default;
    JsonValue& operator=(JsonValue&&) = default;

    // Deep copy
    JsonValue(const JsonValue& o)
        : type(o.type), str_val(o.str_val), num_val(o.num_val), bool_val(o.bool_val) {
        if (o.obj_val) obj_val = std::make_unique<ObjMap>(*o.obj_val);
    }

    int as_int() const { return static_cast<int>(num_val); }
    double as_double() const { return num_val; }
    const std::string& as_string() const { return str_val; }
    bool as_bool() const { return bool_val; }
    const JsonValue& operator[](const std::string& key) const {
        return obj_val->at(key);
    }
    bool has(const std::string& key) const {
        return obj_val && obj_val->count(key) > 0;
    }
};

static void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' ||
                            s[i] == '\r' || s[i] == '\t'))
        i++;
}

static JsonValue parse_json(const std::string& s, size_t& i);

static std::string parse_json_string(const std::string& s, size_t& i) {
    i++; // skip opening "
    std::string result;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') {
            i++;
            if (i < s.size()) result += s[i];
        } else {
            result += s[i];
        }
        i++;
    }
    i++; // skip closing "
    return result;
}

static JsonValue parse_json(const std::string& s, size_t& i) {
    skip_ws(s, i);
    JsonValue v;
    if (s[i] == '"') {
        v.type = JsonValue::STRING;
        v.str_val = parse_json_string(s, i);
    } else if (s[i] == '{') {
        v.type = JsonValue::OBJECT;
        v.obj_val = std::make_unique<JsonValue::ObjMap>();
        i++; // skip {
        skip_ws(s, i);
        while (i < s.size() && s[i] != '}') {
            skip_ws(s, i);
            std::string key = parse_json_string(s, i);
            skip_ws(s, i);
            i++; // skip :
            (*v.obj_val)[key] = parse_json(s, i);
            skip_ws(s, i);
            if (s[i] == ',') i++;
        }
        i++; // skip }
    } else if (s[i] == 't' || s[i] == 'f') {
        v.type = JsonValue::BOOL;
        if (s.substr(i, 4) == "true") { v.bool_val = true; i += 4; }
        else { v.bool_val = false; i += 5; }
    } else {
        // Number
        v.type = JsonValue::NUMBER;
        size_t start = i;
        while (i < s.size() && (std::isdigit(s[i]) || s[i] == '.' ||
                                s[i] == '-' || s[i] == 'e' || s[i] == 'E' ||
                                s[i] == '+'))
            i++;
        v.num_val = std::stod(s.substr(start, i - start));
    }
    return v;
}

static JsonValue load_json_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open " + path);
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    size_t i = 0;
    return parse_json(content, i);
}

// ---------------------------------------------------------------------------
// Memory-mapped file helper
// ---------------------------------------------------------------------------

struct MmapFile {
    void* data = nullptr;
    size_t size = 0;
    int fd = -1;

    ~MmapFile() {
        if (data && data != MAP_FAILED) munmap(data, size);
        if (fd >= 0) close(fd);
    }
};

static MmapFile mmap_open(const std::string& path) {
    MmapFile m;
    m.fd = open(path.c_str(), O_RDONLY);
    if (m.fd < 0) throw std::runtime_error("Cannot open " + path);

    struct stat st;
    fstat(m.fd, &st);
    m.size = st.st_size;

    m.data = mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, m.fd, 0);
    if (m.data == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
    return m;
}

// ---------------------------------------------------------------------------
// BGP query parser
// ---------------------------------------------------------------------------

struct TriplePattern {
    std::string subject;   // URI or "?varname"
    std::string predicate; // URI
    std::string object;    // URI or "?varname"

    bool subject_is_var() const { return !subject.empty() && subject[0] == '?'; }
    bool object_is_var() const { return !object.empty() && object[0] == '?'; }
};

/**
 * Parse a simplified BGP string.
 * Format: "?x <rel1> <entity1> . ?x <rel2> ?y"
 * Triples separated by " . "
 * URIs enclosed in <> or bare, variables start with ?
 */
static std::vector<TriplePattern> parse_bgp(const std::string& query) {
    std::vector<TriplePattern> patterns;

    // Split by " . "
    std::vector<std::string> parts;
    std::string current;
    for (size_t i = 0; i < query.size(); i++) {
        if (query[i] == '.' &&
            (i == 0 || query[i-1] == ' ') &&
            (i+1 >= query.size() || query[i+1] == ' ')) {
            parts.push_back(current);
            current.clear();
            if (i+1 < query.size()) i++; // skip space after dot
        } else {
            current += query[i];
        }
    }
    if (!current.empty()) parts.push_back(current);

    for (auto& part : parts) {
        // Trim
        size_t start = part.find_first_not_of(" \t");
        size_t end = part.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        part = part.substr(start, end - start + 1);

        // Split into 3 tokens
        std::istringstream iss(part);
        TriplePattern tp;
        iss >> tp.subject >> tp.predicate >> tp.object;
        if (tp.subject.empty() || tp.predicate.empty() || tp.object.empty())
            continue;

        // Strip angle brackets from URIs
        auto strip = [](std::string& s) {
            if (s.size() >= 2 && s.front() == '<' && s.back() == '>') {
                s = s.substr(1, s.size() - 2);
            }
        };
        strip(tp.subject);
        strip(tp.predicate);
        strip(tp.object);

        patterns.push_back(tp);
    }
    return patterns;
}

// ---------------------------------------------------------------------------
// Query graph construction
// ---------------------------------------------------------------------------

struct QueryGraph {
    torch::Tensor x;          // (num_nodes, feat_dim)
    torch::Tensor edge_index; // (2, num_edges)
    torch::Tensor edge_attr;  // (num_edges, feat_dim)
    torch::Tensor batch;      // (num_nodes,)
    torch::Tensor graph_size; // (1,)
};

/**
 * Build query graph tensors from parsed BGP patterns.
 *
 * Entity/relation URIs are looked up in the URI-to-ID mapping to retrieve
 * precomputed embeddings and occurrence counts.
 * Variables get zero embeddings and log1p(graph_num_edges) as occurrence counts.
 */
static QueryGraph build_query_graph(
    const std::vector<TriplePattern>& patterns,
    const float* entity_emb, const float* relation_emb,
    const float* entity_occ, const float* relation_occ,
    const std::unordered_map<std::string, int>& entity_uri_to_id,
    const std::unordered_map<std::string, int>& relation_uri_to_id,
    int embedding_dim, int feat_dim, int graph_num_edges)
{
    // Assign local node IDs: collect unique subjects and objects
    std::unordered_map<std::string, int> node_map;
    auto get_node_id = [&](const std::string& name) -> int {
        auto it = node_map.find(name);
        if (it != node_map.end()) return it->second;
        int id = static_cast<int>(node_map.size());
        node_map[name] = id;
        return id;
    };

    std::vector<int64_t> src_nodes, dst_nodes;
    std::vector<int> relation_ids; // index into relation tables, -1 if not found

    for (auto& tp : patterns) {
        int s = get_node_id(tp.subject);
        int o = get_node_id(tp.object);
        src_nodes.push_back(s);
        dst_nodes.push_back(o);

        auto rit = relation_uri_to_id.find(tp.predicate);
        relation_ids.push_back(rit != relation_uri_to_id.end() ? rit->second : -1);
    }

    int num_nodes = static_cast<int>(node_map.size());
    int num_edges = static_cast<int>(patterns.size());

    // Build node features: [occ(3) | embedding(D)]
    auto x = torch::zeros({num_nodes, feat_dim});
    auto x_acc = x.accessor<float, 2>();

    for (auto& [name, local_id] : node_map) {
        bool is_var = (!name.empty() && name[0] == '?');
        if (is_var) {
            // Variable: raw counts = 0, embedding = zero
            // (log1p(0) = 0, applied below)
        } else {
            auto eit = entity_uri_to_id.find(name);
            if (eit != entity_uri_to_id.end()) {
                int eid = eit->second;
                // Occurrence counts
                x_acc[local_id][0] = entity_occ[eid * 3 + 0];
                x_acc[local_id][1] = entity_occ[eid * 3 + 1];
                x_acc[local_id][2] = entity_occ[eid * 3 + 2];
                // Embedding
                for (int d = 0; d < embedding_dim; d++) {
                    x_acc[local_id][3 + d] = entity_emb[eid * embedding_dim + d];
                }
            } else {
                std::cerr << "Warning: entity URI not found: " << name << "\n";
                // Unknown entity: treat as variable (counts = 0)
            }
        }
    }

    // Build edge features: [occ(3) | embedding(D)]
    auto edge_attr = torch::zeros({num_edges, feat_dim});
    auto ea_acc = edge_attr.accessor<float, 2>();

    for (int e = 0; e < num_edges; e++) {
        int rid = relation_ids[e];
        if (rid >= 0) {
            ea_acc[e][0] = relation_occ[rid * 3 + 0];
            ea_acc[e][1] = relation_occ[rid * 3 + 1];
            ea_acc[e][2] = relation_occ[rid * 3 + 2];
            for (int d = 0; d < embedding_dim; d++) {
                ea_acc[e][3 + d] = relation_emb[rid * embedding_dim + d];
            }
        } else {
            std::cerr << "Warning: relation URI not found: "
                      << patterns[e].predicate << "\n";
            // Unknown relation: counts = 0
        }
    }

    // Build edge_index
    auto edge_index = torch::zeros({2, num_edges}, torch::kLong);
    auto ei_acc = edge_index.accessor<int64_t, 2>();
    for (int e = 0; e < num_edges; e++) {
        ei_acc[0][e] = src_nodes[e];
        ei_acc[1][e] = dst_nodes[e];
    }

    // Batch tensor: single query → all zeros
    auto batch = torch::zeros({num_nodes}, torch::kLong);

    // Graph size
    auto graph_size = torch::tensor({static_cast<float>(graph_num_edges)});

    // Apply log1p to occurrence counts (matching training behavior)
    // In training: x_counts = torch.log1p(batch.x[:, :3])
    x.slice(1, 0, 3) = torch::log1p(x.slice(1, 0, 3));
    edge_attr.slice(1, 0, 3) = torch::log1p(edge_attr.slice(1, 0, 3));

    return QueryGraph{x, edge_index, edge_attr, batch, graph_size};
}

// ---------------------------------------------------------------------------
// Run a single query (interactive mode)
// ---------------------------------------------------------------------------

static float run_single_query(
    torch::jit::script::Module& model,
    const std::string& query_string,
    const float* entity_emb, const float* relation_emb,
    const float* entity_occ, const float* relation_occ,
    const std::unordered_map<std::string, int>& entity_uri_to_id,
    const std::unordered_map<std::string, int>& relation_uri_to_id,
    int embedding_dim, int feat_dim, int graph_num_edges)
{
    auto patterns = parse_bgp(query_string);
    auto qg = build_query_graph(
        patterns, entity_emb, relation_emb,
        entity_occ, relation_occ,
        entity_uri_to_id, relation_uri_to_id,
        embedding_dim, feat_dim, graph_num_edges);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(qg.x);
    inputs.push_back(qg.edge_index);
    inputs.push_back(qg.edge_attr);
    inputs.push_back(qg.batch);
    inputs.push_back(qg.graph_size);

    auto output = model.forward(inputs).toTensor();
    return output.item<float>();
}

// ---------------------------------------------------------------------------
// Batch mode: read queries from file, output TSV with per-query timing
// ---------------------------------------------------------------------------

static void run_batch(
    torch::jit::script::Module& model,
    const std::string& batch_file,
    const float* entity_emb, const float* relation_emb,
    const float* entity_occ, const float* relation_occ,
    const std::unordered_map<std::string, int>& entity_uri_to_id,
    const std::unordered_map<std::string, int>& relation_uri_to_id,
    int embedding_dim, int feat_dim, int graph_num_edges)
{
    std::ifstream file(batch_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open batch file: " + batch_file);
    }

    // TSV header
    std::cout << "query_idx\tnum_triples\tlog_cardinality\t"
              << "graph_build_us\tinference_us\ttotal_us\n";

    std::string line;
    int idx = 0;
    torch::NoGradGuard no_grad;

    while (std::getline(file, line)) {
        if (line.empty()) { idx++; continue; }

        auto t_total_start = std::chrono::high_resolution_clock::now();

        auto patterns = parse_bgp(line);
        auto qg = build_query_graph(
            patterns, entity_emb, relation_emb,
            entity_occ, relation_occ,
            entity_uri_to_id, relation_uri_to_id,
            embedding_dim, feat_dim, graph_num_edges);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(qg.x);
        inputs.push_back(qg.edge_index);
        inputs.push_back(qg.edge_attr);
        inputs.push_back(qg.batch);
        inputs.push_back(qg.graph_size);

        auto t_build_done = std::chrono::high_resolution_clock::now();

        auto output = model.forward(inputs).toTensor();

        auto t_total_end = std::chrono::high_resolution_clock::now();

        float log_card = output.item<float>();
        int num_triples = static_cast<int>(patterns.size());
        auto build_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_build_done - t_total_start).count();
        auto infer_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_total_end - t_build_done).count();
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_total_end - t_total_start).count();

        std::cout << idx << "\t" << num_triples << "\t" << log_card << "\t"
                  << build_us << "\t" << infer_us << "\t" << total_us << "\n";
        idx++;
    }

    std::cerr << "Processed " << idx << " queries\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <inference_output_dir> <bgp_query_string>\n"
                  << "       " << argv[0]
                  << " <inference_output_dir> --batch <bgp_file>\n"
                  << "Example: " << argv[0]
                  << " ./output \"?x <rel1> <e1> . ?x <rel2> ?y\"\n";
        return 1;
    }

    std::string output_dir = argv[1];
    bool batch_mode = (argc >= 4 && std::string(argv[2]) == "--batch");

    try {
        // 1. Load metadata
        auto metadata = load_json_file(output_dir + "/metadata.json");
        int embedding_dim = metadata["embedding_dim"].as_int();
        int feat_dim = metadata["feat_dim"].as_int();
        int num_entities = metadata["num_entities"].as_int();
        int num_relations = metadata["num_relations"].as_int();
        int graph_num_edges = metadata["graph_num_edges"].as_int();

        std::cerr << "Graph: " << num_entities << " entities, "
                  << num_relations << " relations, "
                  << graph_num_edges << " edges\n";

        // 2. Load TorchScript model (prefer scripted, fall back to traced)
        torch::jit::script::Module model;
        std::string model_path = output_dir + "/query_gnn_scripted.pt";
        std::ifstream test_file(model_path);
        if (!test_file.good()) {
            model_path = output_dir + "/query_gnn_traced.pt";
        }
        test_file.close();
        std::cerr << "Loading model: " << model_path << "\n";
        model = torch::jit::load(model_path);
        model.eval();

        // 3. Load URI-to-ID mapping
        auto uri_json = load_json_file(output_dir + "/uri_to_id.json");
        std::unordered_map<std::string, int> entity_uri_to_id;
        for (auto& [uri, val] : *uri_json["entities"].obj_val) {
            entity_uri_to_id[uri] = val.as_int();
        }
        std::unordered_map<std::string, int> relation_uri_to_id;
        for (auto& [uri, val] : *uri_json["relations"].obj_val) {
            relation_uri_to_id[uri] = val.as_int();
        }
        std::cerr << "Loaded " << entity_uri_to_id.size() << " entity URIs, "
                  << relation_uri_to_id.size() << " relation URIs\n";

        // 4. Memory-map embeddings and occurrences
        auto ent_emb_mmap = mmap_open(output_dir + "/entity_embeddings.bin");
        auto rel_emb_mmap = mmap_open(output_dir + "/relation_embeddings.bin");
        auto ent_occ_mmap = mmap_open(output_dir + "/entity_occurrences.bin");
        auto rel_occ_mmap = mmap_open(output_dir + "/relation_occurrences.bin");

        auto* entity_emb = static_cast<const float*>(ent_emb_mmap.data);
        auto* relation_emb = static_cast<const float*>(rel_emb_mmap.data);
        auto* entity_occ = static_cast<const float*>(ent_occ_mmap.data);
        auto* relation_occ = static_cast<const float*>(rel_occ_mmap.data);

        if (batch_mode) {
            std::string batch_file = argv[3];
            run_batch(model, batch_file,
                      entity_emb, relation_emb, entity_occ, relation_occ,
                      entity_uri_to_id, relation_uri_to_id,
                      embedding_dim, feat_dim, graph_num_edges);
        } else {
            std::string query_string = argv[2];
            auto patterns = parse_bgp(query_string);
            std::cout << "Parsed " << patterns.size() << " triple patterns\n";
            for (auto& tp : patterns) {
                std::cout << "  " << tp.subject << " " << tp.predicate
                          << " " << tp.object << "\n";
            }

            torch::NoGradGuard no_grad;
            float log_cardinality = run_single_query(
                model, query_string,
                entity_emb, relation_emb, entity_occ, relation_occ,
                entity_uri_to_id, relation_uri_to_id,
                embedding_dim, feat_dim, graph_num_edges);

            float cardinality = std::expm1(log_cardinality);
            if (cardinality < 0) cardinality = 0;

            std::cout << "\nResult:\n";
            std::cout << "  Log-cardinality: " << log_cardinality << "\n";
            std::cout << "  Estimated cardinality: " << cardinality << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
