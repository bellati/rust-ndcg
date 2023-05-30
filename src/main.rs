use std::fs;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let input_file = &args[1];
    let output_file = &args[2];

    let contents =
        fs::read_to_string(input_file)
            .unwrap();

    let mut instances: Vec<Instance> = contents.lines().map(parse_line).collect();
    let instances = instances.as_mut();

    let ndcg_value = calculate_ndcg(instances);

    let _ = fs::write(output_file, ndcg_value.to_string());
}

fn parse_line(line: &str) -> Instance {
    let values: Vec<&str> = line.split_whitespace().collect();
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


#[derive(PartialEq, Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        assert_eq!(parse_line("123 0.93 2 0.82"), Instance { query_id: 123, weight: 0.93, relevancy: 2.0, score: 0.82 });
    }

    #[test]
    #[should_panic]
    fn test_parse_missing_values() {
        parse_line("123 -0.83 2");
    }

    #[test]
    #[should_panic]
    fn test_parse_too_many_values() {
        parse_line("123 -0.83 2 0.82 1.23");
    }

    #[test]
    #[should_panic]
    fn test_parse_negative_weight() {
        parse_line("123 -0.83 2 0.82");
    }

    #[test]
    #[should_panic]
    fn test_parse_zero_weight() {
        parse_line("123 0.0 2 0.82");
    }

    #[test]
    #[should_panic]
    fn test_parse_negative_relevancy() {
        parse_line("123 0.98 -1 0.85");
    }

    #[test]
    fn test_parse_zero_relevancy() {
        assert_eq!(parse_line("123 1.23 0 0.85").relevancy, 0.0);
    }

    #[test]
    fn test_calculate_ndcg_single() {
        let instances = &mut vec![
            Instance { query_id: 0, weight: 0.82, relevancy: 1.0, score: 2.34 },
            Instance { query_id: 0, weight: 1.23, relevancy: 0.0, score: 2.58 }
        ];
        // remember that the weight is defined per query
        //
        // the ndcg should be
        // 1 / log2(3) / 1 = 0.63093
        assert!((calculate_ndcg(instances) - 0.63093).abs() < 0.001);
    }

    #[test]
    fn test_calculate_ndcg() {
        let instances = &mut vec![
            Instance { query_id: 0, weight: 0.82, relevancy: 1.0, score: 2.34 },
            Instance { query_id: 0, weight: 1.23, relevancy: 0.0, score: 2.58 },
            Instance { query_id: 1, weight: 2.56, relevancy: 2.0, score: 1.23 },
            Instance { query_id: 1, weight: 2.46, relevancy: 1.0, score: 0.8  }
        ];
        // remember that the weight is defined per query
        //
        // the first query, the ndcg should be
        // 1 / log2(3) / 1 = 0.63093
        // the second query, the ndcg should be 1 as the order is correct
        //
        // the weight for the queries is
        // 2.05 and 5.02, which totals to 7.07.
        //
        // ndcg should be approximately
        // (2.05 * 0.63093 + 5.02) / 7.07 = 0.8933644
        assert!((calculate_ndcg(instances) - 0.8933644).abs() < 0.001);
    }

    #[test]
    #[should_panic]
    fn test_non_monotonic_query_ids() {
        let instances = &mut vec![
            Instance { query_id: 1, weight: 0.01, relevancy: 1.0, score: 0.8  },
            Instance { query_id: 1, weight: 5.00, relevancy: 2.0, score: 1.23 },
            Instance { query_id: 0, weight: 0.82, relevancy: 1.0, score: 2.34 },
            Instance { query_id: 0, weight: 1.23, relevancy: 0.0, score: 2.58 }
        ];

        calculate_ndcg(instances);
    }

    #[test]
    #[should_panic]
    fn test_only_zero_relevancy() {
        let instances = &mut vec![
            Instance { query_id: 1, weight: 0.01, relevancy: 0.0, score: 0.8  },
            Instance { query_id: 1, weight: 5.00, relevancy: 0.0, score: 1.23 }
        ];

        calculate_ndcg(instances);
    }
}
