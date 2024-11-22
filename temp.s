	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0
	.section	__TEXT,__const
	.p2align	5, 0x0                          ## -- Begin function main
LCPI0_0:
	.quad	0x3ff0000000000000              ## double 1
	.quad	0x4000000000000000              ## double 2
	.quad	0x4000000000000000              ## double 2
	.quad	0x4008000000000000              ## double 3
LCPI0_1:
	.quad	0x4014000000000000              ## double 5
	.quad	0x3ff0000000000000              ## double 1
	.quad	0x401c000000000000              ## double 7
	.quad	0x4020000000000000              ## double 8
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3, 0x0
LCPI0_2:
	.quad	0x4010000000000000              ## double 4
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	4
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movl	$32, %edi
	callq	_malloc
	movq	%rax, %r14
	xorl	%r12d, %r12d
	vmovaps	LCPI0_0(%rip), %ymm0            ## ymm0 = [1.0E+0,2.0E+0,2.0E+0,3.0E+0]
	vmovups	%ymm0, (%rax)
	movl	$32, %edi
	vzeroupper
	callq	_malloc
	movq	%rax, %r15
	vmovapd	LCPI0_1(%rip), %ymm0            ## ymm0 = [5.0E+0,1.0E+0,7.0E+0,8.0E+0]
	vmovupd	%ymm0, (%rax)
	movl	$32, %edi
	vzeroupper
	callq	_malloc
	movq	%rax, %rbx
	xorl	%eax, %eax
	jmp	LBB0_1
	.p2align	4
LBB0_5:                                 ##   in Loop: Header=BB0_1 Depth=1
	incq	%rax
	addq	$16, %r12
LBB0_1:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_4 Depth 2
	cmpq	$1, %rax
	jg	LBB0_6
## %bb.2:                               ## %.preheader4
                                        ##   in Loop: Header=BB0_1 Depth=1
	leaq	(%rbx,%r12), %rcx
	xorl	%edx, %edx
	cmpq	$1, %rdx
	jg	LBB0_5
	.p2align	4
LBB0_4:                                 ##   Parent Loop BB0_1 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movq	$0, (%rcx,%rdx,8)
	incq	%rdx
	cmpq	$1, %rdx
	jle	LBB0_4
	jmp	LBB0_5
LBB0_6:                                 ## %.preheader3
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	jmp	LBB0_7
	.p2align	4
LBB0_14:                                ##   in Loop: Header=BB0_7 Depth=1
	incq	%rcx
	addq	$16, %rax
LBB0_7:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_9 Depth 2
                                        ##       Child Loop BB0_12 Depth 3
	cmpq	$1, %rcx
	jg	LBB0_15
## %bb.8:                               ## %.preheader2
                                        ##   in Loop: Header=BB0_7 Depth=1
	leaq	(%r14,%rax), %rdx
	xorl	%esi, %esi
	xorl	%edi, %edi
	jmp	LBB0_9
	.p2align	4
LBB0_13:                                ##   in Loop: Header=BB0_9 Depth=2
	incq	%rdi
	addq	$8, %rsi
LBB0_9:                                 ##   Parent Loop BB0_7 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_12 Depth 3
	cmpq	$1, %rdi
	jg	LBB0_14
## %bb.10:                              ## %.preheader1
                                        ##   in Loop: Header=BB0_9 Depth=2
	leaq	(%rdi,%rcx,2), %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	cmpq	$1, %r10
	jg	LBB0_13
	.p2align	4
LBB0_12:                                ##   Parent Loop BB0_7 Depth=1
                                        ##     Parent Loop BB0_9 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	vmovsd	(%rdx,%r10,8), %xmm0            ## xmm0 = mem[0],zero
	vmovsd	(%r15,%r9), %xmm1               ## xmm1 = mem[0],zero
	vfmadd213sd	(%rbx,%r8,8), %xmm0, %xmm1 ## xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%rbx,%r8,8)
	incq	%r10
	addq	$16, %r9
	cmpq	$1, %r10
	jle	LBB0_12
	jmp	LBB0_13
LBB0_15:
	vmovsd	LCPI0_2(%rip), %xmm0            ## xmm0 = [4.0E+0,0.0E+0]
	callq	_printFloat
	xorl	%r14d, %r14d
	xorl	%r15d, %r15d
	jmp	LBB0_16
	.p2align	4
LBB0_20:                                ##   in Loop: Header=BB0_16 Depth=1
	incq	%r15
	addq	$16, %r14
LBB0_16:                                ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_19 Depth 2
	cmpq	$1, %r15
	jg	LBB0_21
## %bb.17:                              ## %.preheader
                                        ##   in Loop: Header=BB0_16 Depth=1
	leaq	(%rbx,%r14), %r12
	xorl	%r13d, %r13d
	cmpq	$1, %r13
	jg	LBB0_20
	.p2align	4
LBB0_19:                                ##   Parent Loop BB0_16 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	vmovsd	(%r12,%r13,8), %xmm0            ## xmm0 = mem[0],zero
	callq	_printFloat
	incq	%r13
	cmpq	$1, %r13
	jle	LBB0_19
	jmp	LBB0_20
LBB0_21:
	xorl	%eax, %eax
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	retq
	.cfi_endproc
                                        ## -- End function
.subsections_via_symbols
