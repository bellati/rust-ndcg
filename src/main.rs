use std::fs;

fn main() {
    let filename = "file.txt";

    let contents =
        fs::read_to_string(filename)
            .unwrap();

    let mut instances: Vec<Instance> = contents.lines().map(parse_line).collect();
    let instances = instances.as_mut();

    let ndcg_value = calculate_ndcg(instances);

    println!("{}", ndcg_value);
}

fn parse_line(line: &str) -> Instance {
    let values: Vec<&str> = line.split(' ').collect();
    assert_eq!(values.len(), 4);

    let weight: f32 = values[1].parse().unwrap();
    let relevancy: f32 = values[2].parse().unwrap();
    assert!(weight > 0.0);
    assert!(relevancy >= 0.0);

    return Instance {
        query_id: values[0].parse().unwrap(),
        weight: weight,
        relevancy: relevancy,
        score: values[3].parse().unwrap()
    };
}

fn calculate_ndcg(instances: &mut Vec<Instance>) -> f32 {
    let mut i: usize = 0;
    let mut curr_qid: i32; // curr qid
    let mut size: usize; // number of elements in the current qid

    let mut ndcg_acc: f32 = 0.0;
    let mut weight_acc: f32 = 0.0;

    let num_instances: usize = instances.len();

    // initialization
    if !instances.is_empty() {
        curr_qid = instances[i].query_id;
        size = 1;
    } else {
        return 0.0;
    }

    // loop
    while i < num_instances {

        let next_i = i + 1;
        let start = next_i - size;
        // is last element or next qid is different
        if next_i == num_instances {
            // calculate ndcg
            let ndcg = calculate_query_ndcg(instances, start, next_i);
            ndcg_acc += ndcg.value;
            weight_acc += ndcg.weight;
        } else {
            let next_qid = instances[next_i].query_id;
            assert!(next_qid >= curr_qid);
            // if next element is from a different query
            if next_qid != curr_qid {
                // calculate ndcg
                let ndcg = calculate_query_ndcg(instances, start, next_i);
                ndcg_acc += ndcg.value;
                weight_acc += ndcg.weight;
                // reinitialize
                curr_qid = next_qid;
                size = 1;
            } else {
                size += 1;
            }
        }

        i += 1
    }

    ndcg_acc / weight_acc
}

fn calculate_query_ndcg(instances: &mut Vec<Instance>, start: usize, end: usize) -> WeightedValue {
    assert_ne!(start, end);

    let curr_instances = &mut instances[start..end];

    // orders the current instances by relevancy in descending order
    curr_instances.sort_by(|a, b| b.relevancy.total_cmp(&a.relevancy));

    // calculates idcg
    let idcg = calculate_dcg(curr_instances);
    assert_ne!(idcg.value, 0.0);

    // orders by predicted descending order
    curr_instances.sort_by(|a, b| b.score.total_cmp(&a.score));

    // calculates dcg
    let dcg = calculate_dcg(curr_instances);

    WeightedValue {
        value: dcg.weight * dcg.value / idcg.value,
        weight: dcg.weight
    }
}

// Calculates the Discounted Cumulative Gain score.
fn calculate_dcg(instances: &[Instance]) -> WeightedValue {
    let mut i: f32 = 2.0;
    let mut dcg_acc = 0.0;
    let mut weight_acc = 0.0;
    for instance in instances {
        let num = 2_f32.powf(instance.relevancy) - 1.0;
        let den = i.log2();
        dcg_acc += num / den;
        weight_acc += instance.weight;
        i += 1.0;
    }
    WeightedValue {
        value: dcg_acc,
        weight: weight_acc
    }
}


struct Instance {
    query_id: i32,
    weight: f32,
    relevancy: f32,
    score: f32
}

struct WeightedValue {
    value: f32,
    weight: f32
}
