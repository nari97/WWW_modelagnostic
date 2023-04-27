package RuleExampleCreator;

import org.apache.shiro.crypto.hash.Hash;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Scanner;

public class LogsParser {

    public static HashMap<String, String> get_example_to_entity() throws FileNotFoundException {
        Scanner sc = new Scanner(new File("D:\\PhD\\Work\\EmbeddingInterpretibility\\Interpretibility\\Results\\hetionet_nodes.tsv.txt"));

        HashMap<String, String> example2entity = new HashMap<>();
        while(sc.hasNext()){
            String line = sc.nextLine();
            example2entity.put(line.split("\t")[0], line.split("\t")[1]);
        }

        return example2entity;
    }

    public static String convert_example_to_entity(String example, HashMap<String, String> entity2id){

        example = example.substring(1, example.length()-1);

        String splits[] = example.split(", ");

        String result = "";

        for (String split: splits){
            result += split.substring(0,2) + split.substring(2).split(":")[0] + ":" + entity2id.get(split.substring(2)) + ", ";
        }

        result = result.substring(0, result.length()-2);
        return result;
    }

    public static void parse_logs_and_generate_csvs(String folder_to_parse) throws FileNotFoundException {
        File folder = new File(folder_to_parse);
        File[] files = folder.listFiles();

        HashMap<String, String> example2entity = get_example_to_entity();
        HashMap<String, HashMap<String, String>> all_rules = new HashMap<>();
        String dataset_name = "";
        for(File file: files){
            if (!file.getName().endsWith(".out"))
                continue;
            Scanner sc = new Scanner(file);
            sc.nextLine();
            String line = sc.nextLine();

            String left_side = line.substring(0, line.indexOf("Model"));
            String right_side = line.substring(line.indexOf("Model"));

            dataset_name = left_side.split(":")[1].strip();
            String model_name = right_side.split(":")[1].strip();

            String result = "Rule,Example1,Example2\n";
            while(sc.hasNext()){
                String line1 = sc.nextLine();
                if(line1.contains("Finished")){
                    break;
                }
                String rule = sc.nextLine().replace("RULE:", "");
                sc.nextLine();
                String example1 = convert_example_to_entity(sc.nextLine(), example2entity);
                String example2 = convert_example_to_entity(sc.nextLine(), example2entity);
                result += rule +"\t" + example1 + "\t" + example2 + "\n";

                String predicate = rule.split("==>")[1].substring(0, rule.split("==>")[1].indexOf("("));
                if (!all_rules.containsKey(predicate)){
                    HashMap<String, String> hm = new HashMap<>();
                    all_rules.put(predicate, hm);
                }
                all_rules.get(predicate).put(model_name, rule);
            }
            result = result.substring(0, result.length()-2);
//            System.out.println();
//            System.out.println("Dataset name: " + dataset_name + " Model name: " + model_name);
//            System.out.println(result);

        }
        System.out.println(all_rules);
        String[] model_names = new String[]{"boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"};
        String result = "Predicate";
        for(String model_name: model_names){
            result+="\t" + model_name;
        }
        result += "\n";
        for(String key: all_rules.keySet()){
            result += key;
            for(String model_name: model_names) {
                if(!all_rules.get(key).containsKey(model_name)){
                    result += "\t" + "NULL";
                }
                else{
                    result += "\t" + all_rules.get(key).get(model_name);
                }

            }
            result+="\n";
        }
        result = result.substring(0, result.length()-2);

        System.out.println(result);
    }
    public static void main(String[] args) throws FileNotFoundException {
        String folder_to_parse = "D:\\PhD\\Work\\EmbeddingInterpretibility\\Interpretibility\\Results\\RuleExampleLogs";

        parse_logs_and_generate_csvs(folder_to_parse);
    }
}
