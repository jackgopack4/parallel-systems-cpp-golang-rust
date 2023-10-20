package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type hash_val_idx struct {
	val int
	idx int
}
type ComparisonPair struct {
	ID1 int
	ID2 int
}

type Result struct {
	ID1       int
	ID2       int
	IsEqual   bool
}

type Worker struct {
	ID           int
	Buffer       *ConcurrentBuffer
	ch 					*chan int
}

type ThreadPool struct {
	Workers []*Worker
}

func NewThreadPool(numWorkers, bufferSize int) *ThreadPool {
	pool := &ThreadPool{}
	pool.Workers = make([]*Worker, numWorkers)
	buffer := NewConcurrentBuffer(bufferSize)

	for i := 0; i < numWorkers; i++ {
		worker := &Worker{
			ID:       i + 1,
			Buffer:   buffer,
		}
		pool.Workers[i] = worker
	}

	return pool
}

type ConcurrentBuffer struct {
	lock    sync.Mutex
	readlock sync.Mutex
	notEmpty *sync.Cond
	notFull  *sync.Cond
	numLeftLock sync.Mutex
	buffer  []ComparisonPair
	popindex int
	maxSize int
	numLeft int
}

func NewConcurrentBuffer(maxSize int) *ConcurrentBuffer {
	buffer := &ConcurrentBuffer{
		buffer:  make([]ComparisonPair, 0, maxSize),
		maxSize: maxSize,
	}
	buffer.lock = sync.Mutex{}
	buffer.readlock = sync.Mutex{}
	buffer.numLeftLock = sync.Mutex{}
	buffer.notEmpty = sync.NewCond(&sync.Mutex{})
	buffer.notFull = sync.NewCond(&sync.Mutex{})
	buffer.popindex = 0
	return buffer
}

func (b *ConcurrentBuffer) Push(pair ComparisonPair) {
	//fmt.Println("buffer before push:",b.buffer, "numLeft:",b.numLeft)
	//fmt.Println("called Push for {",pair.ID1,",",pair.ID2,"}")
	b.notFull.L.Lock()
	//fmt.Println("locked b.notFull.L inside Push")
	for len(b.buffer) >= b.popindex + b.maxSize{
		//fmt.Println("waiting for b.notFull")
		b.notFull.Wait()
	}
	//fmt.Println("attempting to lock b.lock inside Push")
	b.lock.Lock()
	//fmt.Println("locked b.lock inside Push")
	b.buffer = append(b.buffer, pair)
	//fmt.Println("buffer after push:",b.buffer,"numLeft:",b.numLeft)
	b.notFull.L.Unlock()
	//fmt.Println("unlocked b.notFull.L inside Push")
	b.lock.Unlock()
	//fmt.Println("unlocked b.lock inside Push")
	b.notEmpty.Signal()
	//fmt.Println("signalled b.notEmpty inside Push")
}

func (b *ConcurrentBuffer) Pop() ComparisonPair {
	//fmt.Println("buffer before Pop:",b.buffer,"numLeft:",b.numLeft)
	//fmt.Println("called Pop")
	b.notEmpty.L.Lock()
	b.readlock.Lock()
	//fmt.Println("Locked notEmpty.L")
	for b.popindex >= len(b.buffer) {
		//fmt.Println("waiting for notEmpty")
		b.notEmpty.Wait()
	}
	//fmt.Println("attempting to lock b.lock inside Pop")

	//fmt.Println("locked b.lock inside Pop")
	pair := b.buffer[b.popindex]
	b.popindex++
	//b.buffer = b.buffer[:len(b.buffer)-1]
	//b.numLeft--
	//fmt.Println("buffer after pop:",b.buffer,"numLeft:",b.numLeft)
	b.notEmpty.L.Unlock()
	//fmt.Println("unlocked b.NotEmpty.L inside Pop")
	b.readlock.Unlock()
	//fmt.Println("unlocked b.lock inside Pop")
	b.notFull.Signal()
	//fmt.Println("signalled b.notFull inside Pop")
	return pair
}

type stack []*TreeNode

func (s stack) Push(v *TreeNode) stack {
    return append(s, v)
}

func (s stack) Pop() (stack, *TreeNode,error) {
    l := len(s)
		if l == 0 {
			return s, nil, errors.New("empty stack")
		}
    return  s[:l-1], s[l-1],nil
}

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
	var print_groups,equal_workers,lock_hashcomp bool
	flag.StringVar(&filename, "filename", "", "string-valued path to an input file")
	flag.IntVar(&hash_workers,"hash-workers", 0, "integer-valued number of threads")
	flag.IntVar(&data_workers,"data-workers", 0, "integer-valued number of threads")
	flag.IntVar(&comp_workers,"comp-workers", 0, "integer-valued number of threads")
	flag.BoolVar(&print_groups,"print-groups",true,"print hash groups and compare groups")
	flag.BoolVar(&equal_workers,"equal-workers",false,"spawn num of goroutines equal to num BSTs")
	flag.BoolVar(&lock_hashcomp,"lock-hashcomp",false,"use lock to protect hashesMap data struct")
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
	hashes_lock:= sync.Mutex{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		inputNumbers = nil
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
		trees = append(trees, bst)
	}
	// fmt.Println("trees:",trees)
	// Calculate hashes
	hashes_map := make(map[int][]int)
	c := make(chan hash_val_idx, len(trees))
	num_hashworkers := calcHashWorkers(hash_workers,equal_workers,len(trees))

	hash_start := time.Now()
	calcHashes(&trees,&hashes_map,num_hashworkers, c,lock_hashcomp,&hashes_lock)
	hash_elapsed := time.Since(hash_start)

	fmt.Println("hashTime:",hash_elapsed.Seconds())
	if comp_workers != 0 || data_workers > 0 {
		hash_group_elapsed := time.Since(hash_start)
		fmt.Println("hashGroupTime:",hash_group_elapsed.Seconds())
		if print_groups {
			for k, indices := range hashes_map {
				if len(indices) > 1 {
					fmt.Print(k,": ")
					for _,v := range indices {
						fmt.Print(v," ")
					}
					fmt.Print("\n")
				}
			}
		}

		compareMap := make(map[int][]int)
		compareMatrix := make([][]bool, len(trees))

		// fmt.Println("compareMatrix:",compareMatrix)
		var compare_start time.Time
		if comp_workers < 0 { // let's do number of workers = number of comparisons
			for i := range compareMatrix {
				compareMatrix[i] = make([]bool, len(trees))
				j := i
				for j < len(compareMatrix[i]) {
					compareMatrix[i][j] = false
					j++
				}
			}
			compare_start = time.Now()
			compTreesMatrix(&compareMatrix, &hashes_map, &trees)
		} else if comp_workers == 1 {
			compare_start = time.Now()
			for _, indices := range hashes_map {
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
		} else if comp_workers > 1 {
			for i := range compareMatrix {
				compareMatrix[i] = make([]bool, len(trees))
				j := i
				for j < len(compareMatrix[i]) {
					compareMatrix[i][j] = false
					j++
				}
			}
			pool := NewThreadPool(comp_workers,comp_workers)
			compare_start = time.Now()
			pool.compTreesWorkers(&compareMatrix,&hashes_map,&trees)
		}
		
		compare_elapsed := time.Since(compare_start)
		fmt.Println("compareTreeTime:",compare_elapsed.Seconds())
		if (comp_workers < 0 || comp_workers > 1) && print_groups {
			fmt.Println("printing comp groups")
			group_idx := 0
			seen := make(map[int]bool)
			for i := range compareMatrix {
				fmt.Println("seeing index",i,"of compareMatrix")
				seen[i] = true
				if !compareMatrix[i][i] {
					j := i+1
					cur := []int{i}
					for j < len(compareMatrix) {
						if compareMatrix[i][j] && !seen[j] {
							seen[j] = true
							cur = append(cur,j)
						}
						j++
					}
					if len(cur) > 1 {
						fmt.Print("group ",group_idx,": ")
						for _,c := range cur {
							fmt.Print(c, " ")
						}
						fmt.Print("\n")
						group_idx += 1
					}
				}
			}
		} else if comp_workers == 1 && print_groups {
			group_idx:=0
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

func compTreesMatrix(matrix *[][]bool, hashes_map *map[int][]int, tree_list *[]TreeNode) {
	var wg sync.WaitGroup
	for _, v := range (*hashes_map) {
		//fmt.Println("ids for curr hash:",v)
		if len(v) == 1 {
			//fmt.Println("v:",v)
			(*matrix)[v[0]][v[0]] = true
		} else {
			for i, t1 := range v {
				j := i+1
				for j < len(v) {
					wg.Add(1)
					t2 := v[j]
					go compTreesParallel(&(*tree_list)[t1], &(*tree_list)[t2],matrix,t1,t2,&wg)
					j += 1
				}
			}
		}
	}
	wg.Wait()
}

func (pool *ThreadPool) compTreesWorkers(matrix *[][]bool, hashes_map *map[int][]int,tree_list *[]TreeNode) {
	bufferSize := len(pool.Workers) // allow for -1 vals at end to close channel
	//var buffer *ConcurrentBuffer
	buffer := NewConcurrentBuffer(bufferSize)
	numCompare := 0
	for _, v := range (*hashes_map) {
		//fmt.Println("ids for curr hash:",v)
		if len(v) == 1 {
			//fmt.Println("v:",v)
			(*matrix)[v[0]][v[0]] = true
		} else {
			for i, _ := range v {
				j := i+1
				for j < len(v) {
					numCompare++
					j += 1
				}
			}
		}
	}
	buffer.numLeftLock.Lock()
	buffer.numLeft = numCompare
	buffer.numLeftLock.Unlock()
	fmt.Println("num to compare:",numCompare)
	var wg sync.WaitGroup
	ch := make(chan int)

	for _, worker := range pool.Workers {
		worker.Buffer = buffer
		worker.ch = &ch
		//buffer.Push(ComparisonPair{ID1: -1, ID2: -1})
	}
	//wg.Add(1)
	//go func() {
		//defer wg.Done()
	//}()


	for _, worker := range pool.Workers {
		wg.Add(1)
		go func(w *Worker) {
			defer wg.Done()
			for {
					/*
					if buffer.numLeft <= i {
						fmt.Println("quitting goroutine",i)
						break
					}
					*/
					pair := buffer.Pop()
					res := compareTrees(&(*tree_list)[pair.ID1],&(*tree_list)[pair.ID2])
					if res {
						(*matrix)[pair.ID1][pair.ID2] = true
					}

					buffer.numLeftLock.Lock()
					buffer.numLeft--
					fmt.Println("buffer:",buffer.buffer)
					if buffer.numLeft < worker.ID {
						buffer.numLeftLock.Unlock()
						break
					}
					buffer.numLeftLock.Unlock()
				}
		}(worker)
	}
	wg.Add(1)
	go func() {
		for _, v := range (*hashes_map) {
			//fmt.Println("ids for curr hash:",v)
			if len(v) == 1 {
				//fmt.Println("v:",v)
				(*matrix)[v[0]][v[0]] = true
			} else {
				//fmt.Println("eq hashes:",v)
				for i, t1 := range v {
					j := i+1
					for j < len(v) {
						//wg.Add(1)
						t2 := v[j]
						fmt.Println("pushing to buffer", t1,t2)
						buffer.Push(ComparisonPair{ID1:t1,ID2:t2})
						j += 1
					}
				}
			}
		}
	}()

	wg.Wait()

}


func compTreesParallel(root1 *TreeNode, root2 *TreeNode, matrix *[][]bool, idx1 int, idx2 int, wg *sync.WaitGroup) {
	defer wg.Done()
	res := compareTrees(root1,root2)
	if res {
		(*matrix)[idx1][idx2] = true
		//(*matrix)[idx2][idx1] = true
	}
}
func calcHashTraversal(root *TreeNode) int {
	hash_val := 1
	q := make(stack,0)
	current := root
	for current != nil || len(q) > 0 {
		for current != nil {
			q = q.Push(current)
			current = current.left
		}

		// Visit the top of the stack
		q, current, _ = q.Pop()
		new_val := current.val + 2;
		hash_val = (hash_val * new_val + new_val) % 1000
		// Move to the right subtree
		current = current.right
	}
	return hash_val
}

func inOrderTraversalIterative(root *TreeNode) []int {
	result := []int{}
	q := make(stack,0)
	current := root
	for current != nil || len(q) > 0 {
		for current != nil {
			q = q.Push(current)
			current = current.left
		}

		// Visit the top of the stack
		q, current, _ = q.Pop()
		result = append(result, current.val)
		// Move to the right subtree
		current = current.right
	}

	return result
}

func calcHashes(tree_list *[]TreeNode, hashes_map *map[int][]int, num_workers int, c chan hash_val_idx, lock_hashcomp bool, lock *sync.Mutex) {
	if num_workers > len(*tree_list) {
		num_workers = len(*tree_list)
	}
	if num_workers <= 1 {
		for idx := range *tree_list {
			hash_val := calcHashTraversal(&(*tree_list)[idx])
			indices, found := (*hashes_map)[hash_val]
			if !found {
				indices = []int{idx}
			} else {
				indices = append(indices,idx)
			}
			(*hashes_map)[hash_val] = indices
		}
	} else {
		var wg sync.WaitGroup
		if lock_hashcomp {
			for i:=0;i<num_workers;i++ {
				wg.Add(1)
				go calcHashesWorkerLock(tree_list,hashes_map,num_workers,i,&wg,lock)
			}
		} else {
			wg.Add(1)
			go hashesMapPutter(hashes_map,len(*tree_list),&wg,c)
			for i:=0;i<num_workers;i++ {
				wg.Add(1)
				go calcHashesWorkerChan(tree_list,num_workers,i, &wg, c) 
			}
		}
		
		wg.Wait()
	}
}

func calcHashesWorkerChan(tree_list *[]TreeNode, num_workers int, idx int, wg *sync.WaitGroup, c chan hash_val_idx) {
	defer wg.Done()
	for idx < len(*tree_list) {
		hash_val := calcHashTraversal(&(*tree_list)[idx])
		c <- hash_val_idx{hash_val,idx}
		idx += num_workers
	}
}

func calcHashesWorkerLock(tree_list *[]TreeNode, hashes_map *map[int][]int, num_workers int, idx int, wg *sync.WaitGroup, lock *sync.Mutex) {
	defer wg.Done()
	for idx < len(*tree_list) {
		hash_val := calcHashTraversal(&(*tree_list)[idx])
		(*lock).Lock()
		indices, found := (*hashes_map)[hash_val]
		if !found {
			indices = []int{idx}
		} else {
			indices = append(indices,idx)
		}
		(*hashes_map)[hash_val] = indices
		(*lock).Unlock()
		idx += num_workers
	}
}

func compareTrees(root1 *TreeNode, root2 *TreeNode) bool {
	list1 := inOrderTraversalIterative(root1)
	list2 := inOrderTraversalIterative(root2)
	if len(list1) != len(list2) {
		return false
	}
	for i := range list1 {		if list1[i] != list2[i] {
			return false
		}
	}
	return true
}

func hashesMapPutter(input *map[int][]int, count int, wg *sync.WaitGroup, c chan hash_val_idx) {
	defer wg.Done()
	for i:=0;i<count;i++ {
		hash_struct := <- c
		hash_val := hash_struct.val
		b_id := hash_struct.idx
		indices, found := (*input)[hash_val]
		if !found {
			indices = []int{b_id}
		} else {
			indices = append(indices,b_id)
		}
		(*input)[hash_val] = indices
	}
}

func calcHashWorkers(hash_workers int, equal_workers bool, length int) int {
	var num_hashworkers int
	if hash_workers == 0 {
		num_hashworkers = 1
	} else if equal_workers {
		num_hashworkers = length
	} else if hash_workers > 10000 {
		num_hashworkers = 10000
	} else {
		num_hashworkers = hash_workers
	}
	return num_hashworkers
}