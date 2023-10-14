package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
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
	flag.StringVar(&filename, "filename", "", "string-valued path to an input file")
	flag.IntVar(&hash_workers,"hash-workers", 1, "integer-valued number of threads")
	flag.IntVar(&data_workers,"data-workers", 1, "integer-valued number of threads")
	flag.IntVar(&comp_workers,"comp-workers", 1, "integer-valued number of threads")
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
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		nums_list := strings.Fields(scanner.Text())
		for _, v := range nums_list {
			num, err := strconv.Atoi(v)
			if err != nil {
				fmt.Println("Error converting to integer:", err)
				return
			}
			inputNumbers = append(inputNumbers, num)
		}
		bst := TreeNode{}

		// Insert numbers into the BST
		for _, num := range inputNumbers {
			bst = *insertNode(&bst,num)
		}

		fmt.Println("Binary Search Tree in-order traversal:")
		hash_val := 1
		inOrderTraversal(&bst, &hash_val)
		fmt.Println()
		fmt.Println("hash_val: ", hash_val)
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
		return
	}
}

func inOrderTraversal(root *TreeNode, hash_val *int) {
	if root != nil {
		inOrderTraversal(root.left, hash_val)
		new_value := root.val + 2;
		*hash_val = (*hash_val * new_value + new_value) % 1000
		fmt.Print(root.val, " ")
		inOrderTraversal(root.right, hash_val)
	}
}