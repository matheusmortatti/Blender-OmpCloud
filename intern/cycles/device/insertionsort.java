import java.util.Random;

public class InsertionSort {
	public static void main(String[] args) {
		int nums = new int(10);
		Random gen = new Random();

		for(int i = 0; i < nums.length; i++) {
			nums[i] = gen.nextInt();
		}

		System.out.println(Arrays.toString(nums));

		insertionsort(nums);
		System.out.println(Arrays.toString(nums));
		
	}

	public void insertionsort(int[] nums) {
		int elect;

		for(int i = 1; i < nums.length; i++) {
			int j = i-1;
			elect = nums[i];

			while(j >= 0 && elect < nums[j]) {
				nums[j] = nums[j+1];
				j--;
			}
			ńums[j+1] = elect;
		}
	}
}