	.text
	.file	"boas_module"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI0_0:
	.quad	0x4024000000000000              # double 10
.LCPI0_1:
	.quad	0x4034800000000000              # double 20.5
.LCPI0_2:
	.quad	0x4049800000000000              # double 51
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rax
	.cfi_def_cfa_offset 16
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = [1.0E+1,0.0E+0]
	movl	$.Lfmt, %edi
	movb	$1, %al
	callq	printf@PLT
	movsd	.LCPI0_1(%rip), %xmm0           # xmm0 = [2.05E+1,0.0E+0]
	movl	$.Lfmt.1, %edi
	movb	$1, %al
	callq	printf@PLT
	movl	$.Lfmt.2, %edi
	movl	$.Lstr, %esi
	xorl	%eax, %eax
	callq	printf@PLT
	movl	$.Lfmt.3, %edi
	movl	$.Lbool, %esi
	xorl	%eax, %eax
	callq	printf@PLT
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = [5.1E+1,0.0E+0]
	movl	$.Lfmt.4, %edi
	movb	$1, %al
	callq	printf@PLT
	xorl	%eax, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	.Lfmt,@object                   # @fmt
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lfmt:
	.asciz	"%.2f\n"
	.size	.Lfmt, 6

	.type	.Lfmt.1,@object                 # @fmt.1
.Lfmt.1:
	.asciz	"%.2f\n"
	.size	.Lfmt.1, 6

	.type	.Lstr,@object                   # @str
.Lstr:
	.asciz	"Boas"
	.size	.Lstr, 5

	.type	.Lfmt.2,@object                 # @fmt.2
.Lfmt.2:
	.asciz	"%s\n"
	.size	.Lfmt.2, 4

	.type	.Lbool,@object                  # @bool
.Lbool:
	.asciz	"true"
	.size	.Lbool, 5

	.type	.Lfmt.3,@object                 # @fmt.3
.Lfmt.3:
	.asciz	"%s\n"
	.size	.Lfmt.3, 4

	.type	.Lfmt.4,@object                 # @fmt.4
.Lfmt.4:
	.asciz	"%.2f\n"
	.size	.Lfmt.4, 6

	.section	".note.GNU-stack","",@progbits
