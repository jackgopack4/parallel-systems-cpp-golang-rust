package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type TreeNode struct {
	val       int
	left, right *TreeNode
}

func insertNode(root *TreeNode, value int) *TreeNode {
	if root == nil {
		return &TreeNode{val: value, left: nil, right: nil}
	}
	if value < root.val {
		root.left = insertNode(root.left, value)
	} else {
		root.right = insertNode(root.right, value)
	}

	return root
}

func main() {
	var filename string
	var hash_workers, data_workers, comp_workers int
	var print_groups,equal_workers bool
	flag.StringVar(&filename, "filename", "", "string-valued path to an input file")
	flag.IntVar(&hash_workers,"hash-workers", 0, "integer-valued number of threads")
	flag.IntVar(&data_workers,"data-workers", 0, "integer-valued number of threads")
	flag.IntVar(&comp_workers,"comp-workers", 0, "integer-valued number of threads")
	flag.BoolVar(&print_groups,"print-groups",true,"print hash groups and compare groups")
	flag.BoolVar(&equal_workers,"equal-workers",false,"spawn num of goroutines equal to num BSTs")
	flag.Parse()
	if filename == "" {
		fmt.Println("Usage: go run filename.go -filename=sample-file.txt")
		return
	}
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	var inputNumbers []int
	var trees []TreeNode
	//var hashes []int
	//idx := 0
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		inputNumbers = nil
		nums_list := strings.Fields(scanner.Text())
		//fmt.Println(nums_list)
		for _, v := range nums_list {
			num, err := strconv.Atoi(v)
			if err != nil {
				fmt.Println("Error converting to integer:", err)
				return
			}
			inputNumbers = append(inputNumbers, num)
		}
		//fmt.Println(inputNumbers)
		bst := TreeNode{}

		// Insert numbers into the BST
		for _, num := range inputNumbers {
			bst = *insertNode(&bst,num)
		}
		trees = append(trees, bst)
		//fmt.Println("Binary Search Tree in-order traversal for tree idx: ",idx)
		
		//idx += 1
	}

	// Calculate hashes
	hashes := make([]int, len(trees))
	var num_hashworkers int
	if hash_workers == 0 {
		num_hashworkers = 1
	} else if equal_workers {
		num_hashworkers = len(trees)
	} else {
		num_hashworkers = hash_workers
	}
	hash_start := time.Now()
	calcHashes(&trees,&hashes,num_hashworkers)
	/*
	for idx, bst := range trees {
		hashes[idx] = 1
		if equal_workers {
			go inOrderTraversal(&bst,&hashes[idx])
		} else {
			inOrderTraversal(&bst, &hashes[idx])
		}
		//fmt.Println()
		//fmt.Println("tree idx: ",idx,"hash_val: ", hash_val)
	}
	*/
	hash_elapsed := time.Since(hash_start)
	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
		return
	}
	fmt.Println("hashTime:",hash_elapsed.Seconds())
	if comp_workers > 0 || data_workers > 0 {
		//hash_group_start := time.Now()
		result := findIndices(hashes)
		//fmt.Println("hash groups:")
		//hash_idx := 0
		hash_group_elapsed := time.Since(hash_start)
		fmt.Println("hashGroupTime:",hash_group_elapsed.Seconds())
		if print_groups {
			for _, indices := range result {
				if len(indices) > 1 {
					fmt.Print(hashes[indices[0]],": ")
					for _,v := range indices {
						fmt.Print(v," ")
					}
					fmt.Print("\n")
				}
			}
		}

		compare_start := time.Now()
		compareMap := make(map[int][]int)
		for _, indices := range result {
			if len(indices) > 1 {
				for i,v := range indices {
					_, found := compareMap[v]
					if !found {
						compareMap[v] = []int{v}
					}
					for j:=i+1; j<len(indices); j++ {
						_, found := compareMap[indices[j]]
						if !found {
							trees_match := compareTrees(&trees[v],&trees[indices[j]])
							if trees_match {
								compareMap[v] = append(compareMap[v],indices[j])
								compareMap[indices[j]] = []int{v}
							}
						}
					} 
				}
			}
		}
		compare_elapsed := time.Since(compare_start)
		fmt.Println("compareTreeTime:",compare_elapsed.Seconds())
		group_idx:=0
		if print_groups {
			for _,groups := range compareMap {
			
				if len(groups) > 1 {
					fmt.Print("group ",group_idx,": ")
					for _,g := range groups {
						fmt.Print(g, " ")
					}
					fmt.Print("\n")
					group_idx += 1
				}
		
			}
		}
	}
}

func inOrderTraversal(root *TreeNode, hash_val *int) []int {
	res := []int{}
	if root != nil {
		res = append(res,inOrderTraversal(root.left, hash_val)...)
		new_value := root.val + 2;
		*hash_val = (*hash_val * new_value + new_value) % 1000
		//fmt.Print(root.val, " ")	
		res = append(res,root.val)
		res = append(res,inOrderTraversal(root.right, hash_val)...)
	}
	return res
}

func calcHashes(tree_list *[]TreeNode, hash_list *[]int, num_workers int) {
	if num_workers > len(*tree_list) {
		num_workers = len(*tree_list)
	}
	if num_workers <= 1 {
		for idx := range *tree_list {
			(*hash_list)[idx] = 1
			inOrderTraversal(&(*tree_list)[idx], &(*hash_list)[idx])
		}
	} else {
		var wg sync.WaitGroup
		for i:=0;i<num_workers;i++ {
			wg.Add(1)
			go calcHashesWorker(tree_list,hash_list,num_workers,i, &wg) 
			
		}
		wg.Wait()
	}
}

func calcHashesWorker(tree_list *[]TreeNode, hash_list *[]int, num_workers int, idx int, wg *sync.WaitGroup) {
	defer wg.Done()
	for idx < len(*tree_list) {
		inOrderTraversal(&(*tree_list)[idx], &(*hash_list)[idx])
		idx += num_workers
	}
}

func compareTrees(root1 *TreeNode, root2 *TreeNode) bool {
	var tmp int
	list1 := inOrderTraversal(root1,&tmp)
	list2 := inOrderTraversal(root2,&tmp)
	if len(list1) != len(list2) {
		return false
	}
	for i := range list1 {		if list1[i] != list2[i] {
			return false
		}
	}
	return true
}

func findIndices(input []int) map[int][]int {
	indexMap := make(map[int][]int)

	for i, value := range input {
		indices, found := indexMap[value]
		if !found {
			indices = []int{i}
		} else {
			indices = append(indices, i)
		}
		indexMap[value] = indices
	}

	return indexMap
}