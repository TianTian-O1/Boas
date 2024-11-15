	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0
	.globl	_main                           ## -- Begin function main
	.p2align	4
_main:                                  ## @main
	.cfi_startproc
## %bb.0:                               ## %entry
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movss	l_matrixA(%rip), %xmm5          ## xmm5 = mem[0],zero,zero,zero
	movss	l_matrixB(%rip), %xmm4          ## xmm4 = mem[0],zero,zero,zero
	movaps	%xmm5, %xmm6
	mulss	%xmm4, %xmm6
	xorps	%xmm3, %xmm3
	addss	%xmm3, %xmm6
	movss	l_matrixA+4(%rip), %xmm1        ## xmm1 = mem[0],zero,zero,zero
	movss	l_matrixB+8(%rip), %xmm2        ## xmm2 = mem[0],zero,zero,zero
	movaps	%xmm1, %xmm0
	mulss	%xmm2, %xmm0
	addss	%xmm6, %xmm0
	movss	%xmm0, 8(%rsp)
	movss	l_matrixB+4(%rip), %xmm6        ## xmm6 = mem[0],zero,zero,zero
	mulss	%xmm6, %xmm5
	addss	%xmm3, %xmm5
	movss	l_matrixB+12(%rip), %xmm7       ## xmm7 = mem[0],zero,zero,zero
	mulss	%xmm7, %xmm1
	addss	%xmm5, %xmm1
	movss	%xmm1, 12(%rsp)
	movss	l_matrixA+8(%rip), %xmm5        ## xmm5 = mem[0],zero,zero,zero
	mulss	%xmm5, %xmm4
	addss	%xmm3, %xmm4
	movss	l_matrixA+12(%rip), %xmm8       ## xmm8 = mem[0],zero,zero,zero
	mulss	%xmm8, %xmm2
	addss	%xmm4, %xmm2
	movss	%xmm2, 16(%rsp)
	mulss	%xmm6, %xmm5
	addss	%xmm3, %xmm5
	mulss	%xmm7, %xmm8
	addss	%xmm5, %xmm8
	movss	%xmm8, 20(%rsp)
	cvtss2sd	%xmm0, %xmm0
	cvtss2sd	%xmm1, %xmm1
	cvtss2sd	%xmm2, %xmm2
	xorps	%xmm3, %xmm3
	cvtss2sd	%xmm8, %xmm3
	leaq	L_format(%rip), %rdi
	movb	$4, %al
	callq	_printf
	xorl	%eax, %eax
	addq	$24, %rsp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__const
	.p2align	2, 0x0                          ## @matrixA
l_matrixA:
	.long	0x3f800000                      ## float 1
	.long	0x00000000                      ## float 0
	.long	0x00000000                      ## float 0
	.long	0x3f800000                      ## float 1

	.p2align	2, 0x0                          ## @matrixB
l_matrixB:
	.long	0x3f800000                      ## float 1
	.long	0x00000000                      ## float 0
	.long	0x00000000                      ## float 0
	.long	0x3f800000                      ## float 1

	.section	__TEXT,__cstring,cstring_literals
L_format:                               ## @format
	.asciz	"Result matrix:\n[%.2f %.2f]\n[%.2f %.2f]\n"

.subsections_via_symbols
